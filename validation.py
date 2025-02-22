from pathlib import Path
from database_connect import FaceVectorModel
from vector_operation import get_face_vector,get_face_label
from database_connect import DatabaseConnect


def validate_result(face_models:list[FaceVectorModel]):
    test = Path.cwd() / "validation_images"
    for file in test.rglob("*"):
        if file.is_dir():
            continue
        target_label = file.parent.name
        vector = get_face_vector(file)
        if vector is None:
            continue
        label, probability = get_face_label(face_models, vector)
        print(target_label, f"预测结果： {label}, {probability}")

db = DatabaseConnect()
results = db.find_all()
faces = [FaceVectorModel(**result) for result in results]

validate_result(faces)