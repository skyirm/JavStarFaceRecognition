import gradio as gr
import faiss
import numpy as np

from vector_operation import get_face_vector_from_array, get_face_label, get_result_from_array
from database_connect import DatabaseConnect, FaceVectorModel

db = DatabaseConnect()



def get_face(img):
    faces = [FaceVectorModel(**face) for face in db.find_all()]
    label = [item.label for item in faces]
    data = [item.vector for item in faces]
    index = faiss.IndexFlatL2(512)
    index.add(np.array(data))

    results = get_result_from_array(img)
    if results is None or len(results) == 0:
        return "未检测到人脸", ""

    for result in results:
        vector = result.normed_embedding
        _, index = index.search(np.array([vector]), 1)

    return get_face_label(faces, vector)


demo = gr.Interface(
    fn=get_face,
    inputs=[gr.Image()],
    outputs=[gr.Textbox(), gr.Number()],
)

demo.launch()