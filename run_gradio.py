from difflib import get_close_matches

import gradio as gr
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from logger import get_logger
from model import FaceVectorModel
from postgresql_connect import PostgresConnection
from vector_operation import get_result_from_array, get_face_vector_from_array, compare_vector

db = PostgresConnection()

logger = get_logger(__name__)

def db_connection_check(func):
    def wrapper(*args, **kwargs):
        if db.db.is_closed():
            logger.warning("数据库连接已断开，尝试重新连接")
            if db.db.connect():
                logger.info("数据库连接成功")
            else:
                logger.error("数据库连接失败")
        return func(*args, **kwargs)
    return wrapper

@db_connection_check
def get_face(img):
    results = get_result_from_array(img)
    if results is None or len(results) == 0:
        return img, "未检测到人脸"

    pil_img = Image.fromarray(np.array(img, dtype=np.uint8))
    portion = int(min(pil_img.size) / 25)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("./font/Alibaba-PuHuiTi-Regular.ttf", size=portion)
    except OSError:
        logger.warn("Cannot find font file")
        font = ImageFont.load_default()

    result_string = ""
    for result in results:
        vector = result.normed_embedding

        name, similarity = db.find_most_similar(vector)

        bbox = result.bbox
        x1, y1, x2, y2 = bbox
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)
        draw.text((x1, y1), name, font=font, fill=(0, 0, 0))
        logger.info("Find %s with similarity %s", name, similarity)
        result_string += f"{name} {similarity:.4f}\n"

    return pil_img, result_string

@db_connection_check
def give_name_suggestion(input_text):
    if not input_text:
        return gr.update(elem_id="name_suggestion", choices=[])
    names = db.find_all_name()

    matches = get_close_matches(input_text, names, n=3, cutoff=0.2)
    if matches:
        return gr.update(elem_id="name_suggestion", choices=matches)
    return gr.update(elem_id="name_suggestion", choices=[])

@db_connection_check
def upload_face(name, img):
    if not name or not img:
        return "人脸和名字不能为空"
    results = get_face_vector_from_array(img)
    if len(results) == 0:
        return "检测不到人脸"
    if len(results) > 1:
        return "检测到多个人脸，无法上传"
    face = FaceVectorModel(vector=results[0], count=1, label=name)
    db.update_one(face)
    logger.info("Upload %s", name)
    return f"上传{name}成功"


def compare_faces(img1, img2):
    if not img1 or not img2:
        return "缺少图片"
    results_1 = get_face_vector_from_array(img1)
    if not results_1:
        return "图片1检测不到人脸"
    if len(results_1) > 1:
        return "图片1有多个人脸"
    results_2 = get_face_vector_from_array(img2)
    if not results_2:
        return "图片2检测不到人脸"
    if len(results_2) > 1:
        return "图片2有多个人脸"
    result = compare_vector(results_1[0], results_2[0])
    return f"相似度{result:2f}"


with gr.Blocks() as demo:
    with gr.Tab("人脸识别"):
        with gr.Row():
            # 输入图片组件
            with gr.Column():
                img_input = gr.Image(label="上传图片", type="pil")
            # 输出图片组件
            with gr.Column():
                img_output = gr.Image(label="结果", type="pil")

        # 文本输出组件
        text_output = gr.Textbox(label="识别结果", lines=3, interactive=False)

        # 处理按钮
        btn = gr.Button("开始识别")
        btn.click(
            fn=get_face,
            inputs=img_input,
            outputs=[img_output, text_output]
        )

    with gr.Tab("上传人脸"):
        img_upload = gr.Image(label="上传图片", type="pil")
        name_upload = gr.Textbox(label="输入名字")
        name_upload_suggestion = gr.Radio(choices=[], label="推荐使用", elem_id="name_suggestion")
        upload_result = gr.Textbox(label="上传结果")
        btn_upload = gr.Button("上传")

        name_upload.input(
            fn=give_name_suggestion,
            inputs=name_upload,
            outputs=name_upload_suggestion,
            trigger_mode="always_last"
        )
        name_upload_suggestion.change(lambda x: x, name_upload_suggestion, name_upload)

        btn_upload.click(
            fn=upload_face,
            inputs=[name_upload, img_upload],
            outputs=upload_result
        )

    with gr.Tab("人脸比对"):
        with gr.Row():
            # 输入图片组件
            with gr.Column():
                img_1 = gr.Image(label="图片1", type="pil")
            # 输出图片组件
            with gr.Column():
                img_2 = gr.Image(label="图片2", type="pil")
        with gr.Row():
            with gr.Column():
                compare_result_output = gr.Textbox(label="比对结果")
            with gr.Column():
                compare_btn = gr.Button("比较")

        compare_btn.click(
            fn=compare_faces,
            inputs=[img_1, img_2],
            outputs=compare_result_output,
        )
logger.info("Start gradio server")
demo.launch(root_path="/gradio", server_port=7860, max_file_size="50MB", show_error=True)
