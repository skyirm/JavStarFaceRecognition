# JavStar Face Recognition

人脸识别服务：AdaFace IR-101（ONNX）+ SCRFD 检测 + SQLite 向量库 + FastAPI 后端 + Svelte 5 前端。

## 架构

```
web/            Svelte 5 + Vite 前端（构建产物 dist/ 由后端托管）
app.py          FastAPI 后端：/api/recognize /api/faces /api/faces/suggest /api/compare
vector_operation.py  SCRFD 检测 + 五点对齐(norm_crop 112) + AdaFace 向量
adaface.py      AdaFace ONNX 推理封装（arena 关闭、单例）
sqlite_connect.py    SQLite 向量库（v2: 一人多向量，检索取 per-name max）
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
   - insightface 模型包 `antelopev2` 放在 `~/.insightface/models/`（只需检测模型）
3. 前端构建（仅开发机需要 Node）：`cd web && npm install && npm run build`
4. 启动：`uv run uvicorn app:app --host 0.0.0.0 --port 7860`
   - 反代子路径部署：`uv run uvicorn app:app --root-path /gradio`

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
