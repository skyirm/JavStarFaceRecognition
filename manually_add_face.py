from pathlib import Path
from random import shuffle

import numpy as np

from model import FaceVectorModel
from postgresql_connect import PostgresConnection
from vector_operation import get_face_vector

db = PostgresConnection()
p = Path(r"E:\国产专区")
file_list = [f for f in p.rglob("*.jpg")]

while True:
    label = input("Enter face label: ")
    vector = np.zeros(512)
    count = 0

    shuffle(file_list)

    for file in file_list:
        if count > 60:
            break
        if file.is_dir():
            continue
        if label in str(file):
            face_vector = get_face_vector(file)
            if face_vector is None:
                continue
            count += 1
            print(file)
            vector += np.array(face_vector)
    mean_vector = vector / count
    if count == 0:
        continue
    face = FaceVectorModel(label=label, vector=mean_vector.tolist(), count=count)
    db.insert_one(face)
    print(f"insert {label} face vector success,count:{count}")

