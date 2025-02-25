import gradio as gr
from vector_operation import get_face_vector_from_array,get_face_label
from database_connect import DatabaseConnect, FaceVectorModel

db = DatabaseConnect()


def get_face(img):
    faces = [FaceVectorModel(**face) for face in db.find_all()]
    vector = get_face_vector_from_array(img)
    if vector is None:
        return "未检测到人脸", ""

    return get_face_label(faces, vector)


demo = gr.Interface(
    fn=get_face,
    inputs=[gr.Image()],
    outputs=[gr.Textbox(), gr.Number()],
)

demo.launch()