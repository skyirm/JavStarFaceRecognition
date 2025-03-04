import gradio as gr
import faiss
from PIL import Image,ImageFont,ImageDraw
import numpy as np

from vector_operation import get_face_vector_from_array, get_face_label, get_result_from_array
from database_connect import DatabaseConnect, FaceVectorModel

db = DatabaseConnect()



def get_face(img):

    results = get_result_from_array(img)
    if results is None or len(results) == 0:
        return img, "未检测到人脸"


    faces = [FaceVectorModel(**face) for face in db.find_all()]
    labels = [item.label for item in faces]
    data = [item.vector for item in faces]

    faiss.normalize_L2(np.array(data,dtype=np.float32))
    index = faiss.IndexFlatL2(512)
    index.add(np.array(data,dtype=np.float32))

    pil_img = Image.fromarray(np.array(img,dtype=np.uint8))
    portion = int(min(pil_img.size)/25)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("./font/Alibaba-PuHuiTi-Regular.ttf", size=portion)
    except:
        font = ImageFont.load_default()

    result_string = ""
    for result in results:
        vector = result.normed_embedding
        vector = vector / np.linalg.norm(vector)

        similarity, position = index.search(np.expand_dims(vector, 0), 1)

        name = labels[position[0][0]]
        # if similarity[0][0] > 0.6:  # 相似度阈值
        #     name = labels[index[0][0]]
        # else:
        #     name = "Unknown"

        bbox = result.bbox
        x1, y1, x2, y2 = bbox

        text_bbox = draw.textbbox((0, 0), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # right - left
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([(x1,y1), (x2,y2)], outline=(255,0,0), width=3)
        draw.text((x1, y1),name,font=font, fill=(0,0,0))
        print(name)
        result_string += f"{name} {similarity[0][0]:.4f}\n"

    return pil_img, result_string


with gr.Blocks() as demo:
    with gr.Row():
        # 输入图片组件
        with gr.Column():
            img_input = gr.Image(label="上传图片", type="numpy")
        # 输出图片组件
        with gr.Column():
            img_output = gr.Image(label="结果")

    # 文本输出组件
    text_output = gr.Textbox(label="识别结果", lines=3, interactive=False)

    # 处理按钮
    btn = gr.Button("开始识别")
    btn.click(
        fn=get_face,
        inputs=img_input,
        outputs=[img_output, text_output]
    )

demo.launch(root_path="/gradio/", server_port=7860)
