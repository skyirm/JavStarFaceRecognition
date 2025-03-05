from pathlib import Path

import numpy as np

from mongo_connect import FaceVectorModel
from postgresql_connect import PostgresConnection
from vector_operation import get_face_vector

db = PostgresConnection()


p = Path.cwd() / "original_images"

face_models = []
for path in p.iterdir():
    label = path.name
    vector = np.zeros(512)
    count = 0
    for file in path.iterdir():
        if count >60:break
        face_vector = get_face_vector(file)
        if face_vector is None:
            continue
        count += 1
        vector += np.array(face_vector)
    mean_vector = vector / count
    face_models.append(FaceVectorModel(mean_vector, label, count))
    db.insert_one(FaceVectorModel(mean_vector.tolist(), label, count))
