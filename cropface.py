from pathlib import Path

from PIL import Image
from mtcnn import MTCNN
from mtcnn.utils.images import load_image

detector = MTCNN()


original_path = Path.cwd() / "original_images"

for origin_f in original_path.rglob("*"):
    file = Path(origin_f)
    if file.is_dir():continue
    image = load_image(str(file))
    results = detector.detect_faces(image)
    if len(results) == 0 or len(results)>1: continue
    x,y,w,h = results[0]['box']
    img = Image.open(str(file))
    cropped_image = img.crop((x,y,x+w,y+h))
    #filename为原父级目录名+原文件名
    target_file = Path.cwd() / "target_images"/ file.parent.name / file.name
    target_file.parent.mkdir(parents=True,exist_ok=True)
    cropped_image.save(str(target_file))


