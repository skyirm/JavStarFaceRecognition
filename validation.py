from pathlib import Path

import numpy as np

from postgresql_connect import PostgresConnection
from vector_operation import get_face_vector_from_file, get_face_label

db = PostgresConnection()

thresholds = list(np.arange(0.2, 1.01, 0.05))
name_list = db.find_all_name()

p= Path().cwd()/"validation_images"

for thresholds in thresholds:
    total_count = 0.
    right_count = 0.
    for file in p.rglob("*.jpg"):
        name = file.parent
        vector = get_face_vector_from_file(file)
        predict_name, score = db.find_most_similar(vector)
        if score > thresholds:
            predict_name = "unknown"
        if name in name_list and predict_name == name:
            right_count += 1
        if name not in name_list and predict_name == "unknown":
            right_count += 1
        total_count += 1

    print(f"thresholds: {thresholds}, right_count: {right_count}, total_count: {total_count}, accuracy: {right_count/total_count}")