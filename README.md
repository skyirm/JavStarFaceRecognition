# JavStar Face Recognition

人脸识别服务：AdaFace IR-101（ONNX）+ SCRFD 检测 + SQLite 向量库 + FastAPI 后端 + Svelte 5 前端。

## 架构

```
web/            Svelte 5 + Vite 前端（构建产物 dist/ 由后端托管）
app.py          FastAPI 后端：/api/recognize /api/faces /api/faces/suggest /api/compare
                /api/faces/{name}/aliases（别名增删） /api/faces/merge（人员合并）
vector_operation.py  SCRFD 检测 + 五点对齐(norm_crop 112) + AdaFace 向量
adaface.py      AdaFace ONNX 推理封装（arena 关闭、单例）
sqlite_connect.py    SQLite 向量库（v2: 一人多向量，检索取 per-name max；
                alias 表支持一人多名，写入时自动解析为主名）
config.py       阈值与路径配置
add_faces.py    从每人一个目录的图片重建人脸库
validation.py   阈值扫描（支持 --baseline 与旧方案对比）
tools/          ONNX 导出脚本、旧 PG 迁移清单工具
```

## 部署（离线可用，无任何远程数据库依赖）

包管理使用 [uv](https://docs.astral.sh/uv/)：

1. 安装依赖（自动创建 .venv）：`uv sync`（生产环境去 dev 组：`uv sync --no-dev`）
   （Windows 上 insightface 需 VS Build Tools 编译）
2. 放置模型文件：
   - `models/adaface_ir101_webface4m.onnx`（来源与导出方法见 `tools/export_adaface_onnx.py` 文档注释）
   - insightface 模型包 `antelopev2` 放在 `~/.insightface/models/`（只需检测模型 `det_10g.onnx`）。
     若解压后多了一层目录，直接把 `antelopev2.zip` 放到 `~/.insightface/models/` 并运行
     `uv run python tools/fix_insightface_models.py` 自动解压/整理，完成后会校验检测模型就位
3. 前端构建（仅开发机需要 Node）：`cd web && npm install && npm run build`

## 启动方法与参数

### 方式一：统一入口 start.py（推荐）

```
uv run python start.py
```

流程：检测 npm → `npm install`（仅首次）→ `npm run build` 构建前端 → 启动 `uvicorn app:app`。
Windows 双击/命令行可用 `start.bat`，POSIX shell 可用 `./start.sh`（均为 start.py 薄封装）。

**start.py 自身参数：**

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--skip-build` | 关 | 跳过前端 npm install/build（要求 `web/dist/` 已存在） |
| `--port` | `7860` | 服务监听端口 |
| `--host` | `0.0.0.0` | 监听地址（与 uvicorn 一致） |
| `--token` | 空 | 管理员令牌，设置后等价于环境变量 `ADMIN_TOKEN` |

**其余参数原样透传给 uvicorn**，常用：

```
# 反代子路径部署（如 http://domain/face 访问，需配合 nginx 透传前缀）
uv run python start.py --skip-build --root-path /face

# 其他 uvicorn 参数同理，如日志级别
uv run python start.py --log-level warning
```

### 方式二：直接 uvicorn（前端已构建好时）

```
uv run uvicorn app:app --host 0.0.0.0 --port 7860
```

等价的还有 `uv run python app.py`（固定 7860 端口）。使用前需保证 `web/dist/` 已构建，
或按第 3 步手动构建。

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ADMIN_TOKEN` | 空 | 设置后 `/api/faces` 管理操作（查看/删除/别名/合并）必须携带 `X-Admin-Token` 请求头；上传、联想、识别、对比不受限；未设置时全站无鉴权 |
| `SQLITE_DB_PATH` | `face_vector.db` | SQLite 数据库文件路径 |

### 反代子路径部署示例

前端请求使用相对路径，部署在 `http://domain/face/` 下无需改代码：

```nginx
rewrite ^/face$ /face/ permanent;        # 强制尾斜杠（前端为相对路径）

location ^~ /face/ {
    client_max_body_size 50m;            # 默认 1m 会挡住图片上传
    proxy_pass http://127.0.0.1:7860/;   # 末尾 / 必须保留：nginx 剥离 /face/ 前缀，
                                         # uvicorn --root-path 会自己拼回去
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

后端启动时追加 `--root-path /face`（路由与文档地址均自动适配）。
注意：uvicorn ≥0.36 的 `--root-path` 要求反代剥离前缀；若 `proxy_pass` 不带末尾 `/`
（原样转发），后端会收到重复前缀（如 `/face/face/`）而 404。

## 鉴权与部署注意

- **库管理鉴权**：见上表 `ADMIN_TOKEN`。仅查看/删除/别名/合并等管理操作需要令牌；
  上传人脸、识别、对比不受限。前端在管理页会自动提示输入令牌。
- **只能单进程**：向量缓存在进程内存，禁止透传 `--workers N`。
- 反代（nginx/caddy）终结 HTTPS；同源托管时无需 CORS。
- 定期备份 `face_vector.db`（连同 -wal/-shm）。

## 建库与验证（从旧 PostgreSQL 迁移）

1. `uv run python validation.py --baseline old.json` 先记录旧方案基线（可选）
2. `uv run python tools/migrate_pg_names.py export --out pg_names.txt`（凭据走环境变量）
3. `uv run python tools/migrate_pg_names.py diff --names pg_names.txt` 输出缺图人员清单，**补齐 original_images 后再建库**
4. `uv run python add_faces.py`（v2：每张注册图一条向量）
5. `uv run python validation.py` 扫阈值，按 `validation_result.json` 确认 `config.SIMILARITY_THRESHOLD`

## 测试

```
uv sync --dev
uv run pytest tests -v
```

## 历史

旧方案：glintr100(ArcFace R100) + 全套辅助模型 + PostgreSQL/pgvector + Gradio。
迁移决策与实施记录见 `docs/refactor-plan.md`。
