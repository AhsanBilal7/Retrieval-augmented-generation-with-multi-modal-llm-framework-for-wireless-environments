from ultralytics import YOLO
from pathlib import Path
import numpy as np
from collections import defaultdict

MODEL_PATH = "path/to/model.pt"
IMAGE_DIR = Path("path/to/images")
OUTPUT_DIR = Path("path/to/output")
PIXELS_PER_METER = 30
FOCAL_LENGTH = 25
OBJECT_HEIGHTS = {
    "car": 1.5,
    "bicycle": 1.0
}

def calculate_distance_from_bottom_center(box, image_width, image_height):
    box_center_x = (box.xyxy[0][0].cpu().numpy() + box.xyxy[0][2].cpu().numpy()) / 2
    box_center_y = box.xyxy[0][3].cpu().numpy()
    bottom_center_x = image_width / 2
    bottom_center_y = image_height
    return np.sqrt((box_center_x - bottom_center_x) ** 2 + (box_center_y - bottom_center_y) ** 2)

def calculate_distance_perspective(box, focal_length, object_height):
    box_height = box.xyxy[0][3] - box.xyxy[0][1]
    return (focal_length * object_height) / (box_height / PIXELS_PER_METER)

model = YOLO(MODEL_PATH)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for image_path in IMAGE_DIR.glob("*.jpg"):
    results = model.predict(str(image_path))
    result = results[0]
    result.save(filename=str(OUTPUT_DIR / f"{image_path.stem}_result.jpg"))
    image_width = result.orig_shape[1]
    image_height = result.orig_shape[0]
    object_distances = defaultdict(list)
    for box in result.boxes:
        object_type = model.names[int(box.cls[0])]
        distance_center_px = calculate_distance_from_bottom_center(box, image_width, image_height)
        distance_center_m = distance_center_px / PIXELS_PER_METER
        if object_type in OBJECT_HEIGHTS:
            object_height = OBJECT_HEIGHTS[object_type]
            distance_perspective = calculate_distance_perspective(box, FOCAL_LENGTH, object_height)
            object_distances[object_type].append(distance_perspective)
        else:
            object_distances[object_type].append(distance_center_m)
    description_lines = [f"Image: {image_path.name}"]
    for object_type, distances in object_distances.items():
        distances_str = ", ".join([f"{d:.2f} meters" for d in distances])
        description_lines.append(f"{len(distances)} {object_type}(s) at {distances_str}")
    description = "\n".join(description_lines)
    print(description)
    with open(OUTPUT_DIR / f"{image_path.stem}_description.txt", "w") as desc_file:
        desc_file.write(description)
