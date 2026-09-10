from pathlib import Path
import insightface
import cv2
import numpy as np

from vector_operation import get_face_vector_from_file


def drop_image():
    p = Path().cwd()
    for file in p.rglob("*.jpg"):
        if file.is_dir():
            continue
        face = get_face_vector_from_file(str(file))
        if face is None:
            print(f"drop {file}")
            file.unlink()


drop_image()
