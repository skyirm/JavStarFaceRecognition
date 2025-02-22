from pathlib import Path

import numpy as np

from vector_operation import get_face_vector


label = ""
p = Path(r"E:\国产专区")

vector = np.zeros(512)
count = 0

for file in p.rglob("*.jpg"):
    if file.is_dir():
        continue
    if label in file.parts:
        face_vector = get_face_vector(file)
        if face_vector is None:
            continue
        count += 1
        vector += np.array(face_vector)
mean_vector = vector / count
