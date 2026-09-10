# 修改方案：识别模型升级 AdaFace IR-101 + 数据库迁移 SQLite

> 状态：**已定稿**（讨论日期：2026-09-11，全部待定项已拍板）
> 范围：人脸识别服务（后端 FastAPI + 前端 Svelte，及配套脚本）

---

## 1. 背景与问题

当前架构：

| 模块 | 现状 |
|---|---|
| 检测+对齐 | insightface `antelopev2`（SCRFD det_10g），同时加载了 genderage、双 landmark 等无用模型 |
| 识别 | glintr100（ArcFace R100，fp32 约 250MB） |
| 存储 | 远程 PostgreSQL + pgvector（peewee ORM，`cosine_distance` 检索） |
| 界面 | Gradio（端口 7860，并发 2） |

主要问题：

1. **内存吃满**：识别模型 + 全套辅助模型 + ORT 内存 arena 只增不减 + 大图解码数组多次拷贝 + 并发 2。
2. **外部依赖重**：每次比对都要跨网络访问远程 PostgreSQL（延迟、可用性、凭据明文硬编码）。
3. **精度诉求**：希望进一步提升识别准确率。

---

## 2. 目标与非目标

**目标**
- 识别模型升级为 AdaFace IR-101（开源高精度方案）。
- 存储迁移为本地 SQLite 单文件，去除 PostgreSQL 依赖。
- 架构改为前后端分离：后端 FastAPI（REST API），前端 Svelte 5（Vite 构建），替换 Gradio。
- 新增人脸库管理页（浏览/搜索/删除已有人员）。
- 顺带完成内存优化与历史代码清理。

**非目标**
- 不做分布式/亿级向量检索（当前为封闭集，量级远未到）。

---

## 3. 总体方案

### 3.1 模型方案

> 说明：AdaFace IR-101 是**识别**模型（把对齐后的人脸变成向量），不负责检测人脸。
> 检测/对齐保持现有 SCRFD det_10g 不动（检测端换模型收益很小），只替换识别端。

**权重来源（已定）**：从官方仓库（mk-minchul/AdaFace，MIT）WebFace4M 预训练 `.pth` 自行导出 ONNX（一次性，仅开发机需 torch），导出后用同一批对齐图做 torch vs onnxruntime 数值对拍（余弦 ≥ 0.999）。**本期不做 int8 量化**，fp32 先行。

| 环节 | 方案 |
|---|---|
| 检测+对齐 | insightface FaceAnalysis，`allowed_modules=["detection"]`，只加载 det_10g |
| 识别 | AdaFace IR-101 ONNX，直接用 onnxruntime 推理，输出 512 维向量后 L2 归一化 |
| 对齐输入 | 复用 insightface 的五点对齐 `norm_crop(img, face.kps, size=112)`，与 AdaFace 官方仓库的输入方式一致 |
| 预处理 | 112×112、归一化 `(x/255 - 0.5) / 0.5`（**需对照官方仓库逐行核对 BGR/RGB 与数值范围**，见待讨论 #2） |

**收益**
- 精度：AdaFace（WebFace4M 预训练）在 IJB-C 等基准上超过 ArcFace R100，对小脸/低质量图鲁棒性更好。
- 内存：`allowed_modules=["detection"]` 后不再加载 glintr100（-250MB）及 genderage/landmark（约 -10MB）；AdaFace IR-101 fp32 约 250MB，整体内存基本持平或略降，精度显著提升。

### 3.2 数据库方案（SQLite）

单文件本地库 + numpy 全量暴力检索（封闭集几千人以内毫秒级，且省去网络往返）。

```sql
CREATE TABLE IF NOT EXISTS face_vector (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name   TEXT NOT NULL,
    count  INTEGER NOT NULL DEFAULT 1,
    vector BLOB NOT NULL              -- float32[512] 的 tobytes()
);
CREATE INDEX IF NOT EXISTS idx_face_vector_name ON face_vector(name);
```

- 写入：`np.asarray(vec, dtype=np.float32).tobytes()`；读取：`np.frombuffer(blob, dtype=np.float32)`。
- 检索：启动时全量载入 `(N, 512)` 矩阵并按行归一化，`sims = M @ v` 一次点积得到所有相似度；insert/update/delete 时同步维护该矩阵；v2 下向量行数 > 人数，属预期。
- **本期直接实现 v2（一人多向量，已定）**：同一 name 多行，`find_most_similar` 按 name 分组取组内最大相似度。`add_faces.py` 每张注册图单独存一行向量，不再求均值（v1 均值方案废弃）；删除人员时移除其全部行并重建缓存矩阵。

### 3.3 相似度语义统一（顺带修一个 bug）

现状三处语义混乱（均为"距离"语义：越小越相似）：
- `postgresql_connect.find_most_similar` 用 pgvector `cosine_distance`，返回距离；
- `run_gradio.py:40` 用 `similarity > 0.5 → Unknown`（距离语义，正确）；
- `vector_operation.get_face_label` 取 `max_similarity` 当最优（**bug**：距离应该取 min，目前无人调用）。

**本次统一为"相似度"语义（越大越相似）**：
- `compare_vector` 改为直接返回余弦相似度 `dot(a, b)`（向量已归一化）；
- `find_most_similar` 返回 `(name, max_similarity)`；
- `run_gradio.py` 判断改为 `if similarity < SIMILARITY_THRESHOLD: name = "Unknown"`；
- `validation.py` 的扫阈值方向同步反转。

### 3.4 前端方案（已定）

| 层 | 方案 |
|---|---|
| 后端 | FastAPI + uvicorn 单 worker；模型/DB 在 lifespan 中初始化；semaphore 限并发 1；`--root-path` 兼容现有反代子路径部署；比 Gradio 省几百 MB 内存 |
| 前端 | Svelte 5 + Vite，`vite build` 产出 `dist/`，由 FastAPI StaticFiles 托管；开发机需 Node，部署机不需要 |
| 结果标注 | 后端只返回 JSON（name / similarity / bbox），识别框和文字由前端 canvas 绘制，浏览器字体直接渲染中文，免去后端 PIL 字体依赖 |

API 设计：

```
POST   /api/recognize        multipart 图片 → [{name, similarity, bbox: [x1, y1, x2, y2]}]
POST   /api/faces            multipart name + 图片 → 校验单人脸/置信度后入库
GET    /api/faces            人脸库列表（name/count），管理页用
DELETE /api/faces/{name}     删除某人（向量矩阵同步移除）
GET    /api/faces/suggest?q= 名称建议（get_close_matches）
POST   /api/compare          两张图 → {similarity}
```

页面：① 人脸识别 ② 上传人脸（含名称建议）③ 人脸比对 ④ 人脸库管理（浏览/搜索/删除）。

---

## 4. 改动清单

| 文件 | 改动 |
|---|---|
| `vector_operation.py` | detector 限 `allowed_modules=["detection"]`；新增 AdaFace 加载；`get_face_vector_from_file/array` 改为 det → norm_crop → AdaFace；`compare_vector` 改相似度语义；修 `get_face_label` 或删除 |
| `adaface.py`（新增） | AdaFace ONNX 封装：SessionOptions 关闭 CPU arena、单例加载、`get(aligned_img) -> normed_vec` |
| `sqlite_connect.py`（新增） | 接口对齐 `PostgresConnection` 并扩展：`insert_one / update_one / delete_one_by_name / find_one_by_name / find_most_similar / find_all_name`；向量矩阵缓存（增/删时同步维护）；`check_connection` 保留为空操作装饰器 |
| `app.py`（新增，替代 run_gradio.py） | FastAPI：实现 §3.4 全部 API；lifespan 加载模型/DB；semaphore 限并发 1；上传图片长边 >1280 时先缩放；Unknown 判断方向反转；StaticFiles 托管前端 dist。`run_gradio.py` 删除，gradio 依赖移除 |
| `web/`（新增） | Svelte 5 + Vite 前端工程：人脸识别 / 上传人脸 / 人脸比对 / 人脸库管理 四个页面 |
| `add_faces.py` | 换用 sqlite_connect + model.FaceVectorModel（当前从 `mongo_connect` 导入，属历史遗留）；v2：每张注册图单独存一行向量，不再求均值 |
| `manually_add_face.py` | 同上换 import |
| `validation.py` | 适配新 DB 与相似度语义；输出新旧方案对比表 |
| `config.py` | 新增 `ADAFACE_ONNX_PATH`、`SIMILARITY_THRESHOLD`；删除 MongoDB 连接串与数据库明文密码 |
| `mongo_connect.py` | **删除**（确认无引用后） |
| `postgresql_connect.py` | 保留只读作为迁移基线，迁移完成后删除 |
| `tools/migrate_pg_names.py`（新增） | 只导出 PG 中的 name+count 作为清单，用于和 original_images 比对完整性（**向量本身不可迁移**，见 §5） |

---

## 5. 数据迁移

**关键点：旧向量（glintr100 空间）与新向量（AdaFace 空间）不兼容，pgvector 数据不能直接搬迁，必须重算。**

1. 迁移前先用 `validation.py` 跑一轮旧方案，把准确率基线记录下来。
2. 用 `tools/migrate_pg_names.py` 导出 PG 里的全部 name，与 `original_images/` 目录名比对：
   - 完全覆盖 → 直接重跑 `add_faces.py`（新模型）建库；
   - 有缺口 → **等补图后再建库（已定）**：脚本输出缺图人员清单，人工补齐 `original_images` 后才执行建库，避免带缺口上线。
3. 重跑 `add_faces.py`，生成 `face_vector.db`。
4. 部署只带一个 db 文件 + ONNX 模型文件，不再依赖远程数据库。

---

## 6. 验证与验收标准

- `validation.py` 扫阈值：新方案准确率 ≥ 旧方案基线，并确定 `SIMILARITY_THRESHOLD`（AdaFace 同人余弦通常 >0.6，非同人 <0.4，实际以扫出来为准）。
- 内存：服务空闲 RSS ≤ 1.5GB，识别峰值 ≤ 2GB（CPU 服务器，目标值待确认）。
- 性能：单图（≤1280 长边）端到端识别 CPU 耗时 ≤ 3s（AdaFace IR-101 单张约 100~300ms，可接受）。
- 部署：断网状态下完整功能可用（无任何远程 DB 依赖）。

---

## 7. 决策记录（原待讨论议程）

> 全部已拍板（2026-09-11 讨论）：

1. **ONNX 权重来源**：官方 `.pth` 自行导出 + 数值对拍验证；本期不做 int8，fp32 先行。
2. **预处理对齐**：逐行对照官方 demo；导出后 torch vs onnxruntime 对拍兜底（余弦 ≥ 0.999）。
3. **相似度语义**：统一为"越大越相似"，run_gradio(app) / validation / find_most_similar 三处同步改。
4. **存储**：本期直接实现 v2（一人多向量、取 max），v1 均值方案废弃。
5. **缺图人员**：等补图后再建库，`migrate_pg_names.py` 输出缺口清单，人工补齐后才执行建库。
6. **代码清理**：删 `mongo_connect.py`、移除明文密码；密码已进 git 历史，需人工轮换（数据库侧操作）。
7. **内存优化**：并入本期（大图缩放、并发 1、ORT arena 关闭）。
8. **旧 PostgreSQL**：迁移验收通过后 `pg_dump` 备份留本地，随后下线连接配置。
9. **前端架构**：FastAPI + Svelte 5（Vite 构建）前后端分离，替换 Gradio；新增人脸库管理页（浏览/搜索/删除）。

---

## 8. 工作量估算

| 事项 | 估算 |
|---|---|
| adaface.py + vector_operation.py 改造 | 0.5 天（含权重获取/验证） |
| sqlite_connect.py | 0.5 天 |
| add_faces / validation / run_gradio 适配 + 联调 | 0.5 天 |
| 建库 + 扫阈值 + 新旧对比 | 0.5 天 |
| 前端 Svelte 工程（4 页面）+ API 联调 | 1 天 |

合计约 3 个工作日（权重来源顺利的前提下）。
