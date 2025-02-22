from pathlib import Path
import insightface
import cv2
import numpy as np

detector = insightface.app.FaceAnalysis(
    name="buffalo_l", providers=['CUDAExecutionProvider',"CPUExecutionProvider"]
)
detector.prepare(ctx_id=0, det_size=(640, 640))


def drop_image():
    p = Path().cwd()
    for file in p.rglob("*.jpg"):
        if file.is_dir():
            continue
        img = cv2.imdecode(np.fromfile(file, np.uint8), cv2.IMREAD_COLOR)
        faces = detector.get(img)
        if len(faces) == 0:
            file.unlink()
            print(f"{file} has been deleted.")
        if len(faces) > 1:
            file.unlink()
            print(f"{file} has been deleted.")


drop_image()
