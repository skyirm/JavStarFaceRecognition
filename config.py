# 检测置信度阈值：低于此值的人脸不用于注册
FACE_DETECT_THRESHOLD = 0.7

# 识别相似度阈值（越大越相似）：最高相似度低于此值判定为 Unknown
# 初始值 0.5，以 validation.py 扫描结果为准
SIMILARITY_THRESHOLD = 0.5

# 每张脸返回的候选人数
RECOGNIZE_TOP_K = 3

# AdaFace IR-101 ONNX 模型路径（相对项目根目录）
ADAFACE_ONNX_PATH = "models/adaface_ir101_webface4m.onnx"

# SQLite 数据库文件路径
SQLITE_DB_PATH = "face_vector.db"

# 识别历史最多保留条数（FIFO 淘汰最旧的）
RECOGNIZE_LOG_MAX = 5000

# 库体检：跨名向量相似度 >= 此值时提示疑似同一人（可合并）
CLUSTER_MERGE_THRESHOLD = 0.6

# 库体检：同名向量与本人均值向量的相似度 < 此值时提示离群（疑似误注册）
CLUSTER_OUTLIER_THRESHOLD = 0.5
