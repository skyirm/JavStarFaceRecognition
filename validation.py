from pathlib import Path

import numpy as np

from postgresql_connect import PostgresConnection
from vector_operation import get_face_vector_from_file, get_face_label

db = PostgresConnection()

thresholds = list(np.arange(0.1, 1.01, 0.05))
name_list = db.find_all_name()

p= Path().cwd()/"validation_images"

for threshold in thresholds:
    total_count = 0.
    right_count = 0.
    for file in p.rglob("*.jpg"):
        name = file.parent.name
        vector = get_face_vector_from_file(file)
        if vector is None:
            continue
        find_name, score = db.find_most_similar(vector)
        predict_name = find_name if score <threshold else "unknown"
        # print(f"{score}{'>' if score>threshold else '<'}{threshold} {name} {find_name}->{predict_name}",end=" ")
        if name in name_list and predict_name == name:
            right_count += 1
            # print("+1")
        if name not in name_list and predict_name == "unknown":
            right_count += 1
            # print("+1")
        # print("")
        total_count += 1

    print(f"threshold: {threshold}, right_count: {right_count}, total_count: {total_count}, accuracy: {right_count/total_count}")