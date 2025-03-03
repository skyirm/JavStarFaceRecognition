import gradio as gr
import faiss
from PIL import Image,ImageFont,ImageDraw
import numpy as np

from vector_operation import get_face_vector_from_array, get_face_label, get_result_from_array
from database_connect import DatabaseConnect, FaceVectorModel

db = DatabaseConnect()



def get_face(img):
    faces = [FaceVectorModel(**face) for face in db.find_all()]
    labels = [item.label for item in faces]
    data = [item.vector for item in faces]

    faiss.normalize_L2(np.array(data,dtype=np.float32))
    index = faiss.IndexFlatL2(512)
    index.add(np.array(data,dtype=np.float32))

    pil_img = Image.fromarray(np.array(img,dtype=np.uint8))
    portion = int(min(pil_img.size)/30)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("./font/fzheiti.ttf", size=portion)
    except:
        font = ImageFont.load_default()

    results = get_result_from_array(img)
    if results is None or len(results) == 0:
        return "未检测到人脸", ""

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
        draw.text((x1, y1),name,font=font, fill=(255,0,0))
        print(name)

    return pil_img


demo = gr.Interface(
    fn=get_face,
    inputs=[gr.Image(type="pil")],
    outputs=[gr.Image(type="pil")],
)

demo.launch()
