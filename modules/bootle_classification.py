import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import tempfile
import json
import datetime
import os

# # Safe load patch untuk PyTorch 2.6+ jika pakai Ultralytics
# try:
#     import ultralytics.nn.modules.conv as conv_mod
#     import ultralytics.nn.modules.head as head_mod
#     import ultralytics.nn.modules.block as block_mod
#     import ultralytics.nn.tasks as tasks_mod
#     import torch.nn.modules.container

#     torch.serialization.add_safe_globals([
#         tasks_mod.DetectionModel,
#         conv_mod.Conv,
#         head_mod.Detect,
#         block_mod.C2f,
#         torch.nn.modules.container.Sequential
#     ])
# except Exception as e:
#     st.warning(f"Gagal patch torch globals (boleh diabaikan jika bukan PyTorch 2.6+): {e}")

# Load model
model_path = "./ml_models/defect_classification/best.pt"
if not os.path.exists(model_path):
    st.error("Model tidak ditemukan. Pastikan file 'best.pt' ada di path: ./ml_models/defect_classification/")
    st.stop()

from ultralytics import YOLO
model = YOLO(model_path)

def run():
    st.header("Klasifikasi Botol: PROPER / DEFECT")
    uploaded_file = st.file_uploader("Upload gambar botol", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        # Simpan sementara gambar asli untuk inferensi
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = model(tmp.name, imgsz=960, conf=0.1)

        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        # Status default dan threshold confidence minimum
        status = {"Cap": False, "Label": False, "water_level": False, "Bottle": False, "bad_label": False}
        thresholds = {"Cap": 0.01, "Label": 0.1, "water_level": 0.8, "Bottle": 0.6, "bad_label": 0.1}
        confidence = {}

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                text = f"{label} {conf:.2f}"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                draw.rectangle([(x1, y1 - text_height), (x1 + text_width, y1)], fill="red")
                draw.text((x1, y1 - text_height), text, fill="white", font=font)

                if label in thresholds:
                    if conf >= thresholds[label]:
                        status[label] = True
                    confidence[label] = conf

        required_keys = ["Cap", "Label", "water_level", "Bottle"]
        final = "PROPER" if all(status.get(k, False) for k in required_keys) else "DEFECT"

        # Resize tampilan
        resized_img = img_with_boxes.resize((300, int(img_with_boxes.height * 300 / img_with_boxes.width)))

        col1, col2 = st.columns(2)
        with col1:
            st.image(resized_img, caption="Deteksi Botol")

        with col2:
            st.subheader(f"Hasil: **{final}**")
            st.write("Status Komponen:", status)
            st.write("Confidence Score:")
            for label, conf in confidence.items():
                st.write(f"- {label}: {conf:.2f}")

            # Simpan hasil
            json_file = save_data(status, confidence)
            st.success("Hasil disimpan ke JSON.")

def save_data(status_dict, confidence_dict):
    folder_path = "database_json"
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, "hasil_deteksi_list.json")

    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    id_data = max([d.get("id", 0) for d in existing_data], default=0) + 1
    time_checked = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "id": id_data,
        "Cap": status_dict.get("Cap", False),
        "Label": status_dict.get("Label", False),
        "water_level": status_dict.get("water_level", False),
        "Bottle": status_dict.get("Bottle", False),
        "bad_label": status_dict.get("bad_label", False),
        "confidence": {k: round(v, 4) for k, v in confidence_dict.items()},
        "date_checked": time_checked
    }

    existing_data.append(data)
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

    return filename
