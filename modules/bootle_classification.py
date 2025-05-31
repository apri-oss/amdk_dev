import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tempfile
import json
import datetime
import os
from ultralytics import YOLO

model = YOLO("./ml_models/defect_classification/best.pt")

def run():
    st.header("Klasifikasi Botol PROPER / DEFECT")
    uploaded_file = st.file_uploader("Upload gambar botol", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)

        # Simpan sementara gambar asli untuk inferensi
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = model(tmp.name, imgsz=960, conf=0.1)

        # Copy gambar untuk gambar bounding box
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        status = {"Cap": False, "Label": False, "water_level": False, "Bottle": False, "bad_label": False}
        thresholds = {"Cap": 0.01, "Label": 0.1, "water_level": 0.8, "Bottle": 0.6, "bad_label": 0.1}
        confidence = {}

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id]

                # koordinat bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # teks label + confidence
                text = f"{label} {conf:.2f}"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # gambar kotak dan teks
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                draw.rectangle([(x1, y1 - text_height), (x1 + text_width, y1)], fill="red")
                draw.text((x1, y1 - text_height), text, fill="white", font=font)

                if label in thresholds:
                    if conf >= thresholds[label]:
                        status[label] = True
                    confidence[label] = conf

        required_keys = ["Cap", "Label", "water_level", "Bottle"]
        final = "PROPER" if all(status.get(k, False) for k in required_keys) else "DEFECT"

        # Resize untuk tampilan
        new_width = 300
        new_height = int(img_with_boxes.height * new_width / img_with_boxes.width)
        resized_img = img_with_boxes.resize((new_width, new_height))

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(resized_img, caption="Gambar dengan Bounding Box")

        with col2:
            st.markdown(f"### HASIL: **{final}**")
            st.write("Detail Komponen:", status)

            st.write("Confidence Score:")
            for label, conf in confidence.items():
                st.write(f"- {label}: {conf:.2f}")

            json_file = save_data(status, confidence)
            st.success("Hasil deteksi disimpan.")

def save_data(status_dict, confidence_dict):
    folder_path = "database_json"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = os.path.join(folder_path, "hasil_deteksi_list.json")

    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    if existing_data:
        id_data = max(item.get("id", 0) for item in existing_data) + 1
    else:
        id_data = 1

    time_checked = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "id": id_data,
        "Cap": status_dict.get("Cap", False),
        "Label": status_dict.get("Label", False),
        "water_level": status_dict.get("water_level", False),
        "Bottle": status_dict.get("Bottle", False),
        "bad_label": status_dict.get("bad_label", False),
        "confidence": {label: round(conf, 4) for label, conf in confidence_dict.items()},
        "date_checked": time_checked
    }

    existing_data.append(data)

    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

    return filename
