import sys
import json
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("ml_models/defect_classification/best.pt")

def main():
    payload = json.load(sys.stdin)
    image_path = payload["image_path"]
    thresholds = payload.get("thresholds", {})

    # Inferensi
    results = model(image_path, imgsz=960, conf=thresholds.get("Label", 0.1))
    status = {"Cap": False, "Label": False, "water_level": False, "Bottle": False, "bad_label": False}

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf  = float(box.conf[0])
            label = r.names[cls_id]
            if label in thresholds and conf >= thresholds[label]:
                status[label] = True

    required = ["Cap", "Label", "water_level", "Bottle"]
    status["final"] = "PROPER" if all(status[k] for k in required) else "DEFECT"

    # Cetak JSON
    print(json.dumps(status))

if __name__ == "__main__":
    main()
