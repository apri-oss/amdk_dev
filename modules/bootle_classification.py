import streamlit as st
import torch
from PIL import Image
import tempfile
import json
import datetime
import os
# Import ultralytics modules
import ultralytics.nn.tasks
import ultralytics.nn.modules.conv
import torch.nn.modules.container

# Allowlist class yang dibutuhkan untuk torch.load
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    ultralytics.nn.modules.conv.Conv
])

from ultralytics import YOLO
model = YOLO("./ml_models/defect_classification/best.pt")

def run():
    st.header("Klasifikasi Botol PROPER / DEFECT")
    uploaded_file = st.file_uploader("Upload gambar botol", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        new_width = 300
        new_height = int(img.height * new_width / img.width)
        resized_img = img.resize((new_width, new_height))
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(resized_img, caption="Gambar yang diunggah")
        with col2:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name)
                results = model(tmp.name, imgsz=960, conf=0.1)
            status = {"Cap": False, "Label": False, "water_level": False, "Bottle": False, "bad_label": False}
            thresholds = {"Cap": 0.01, "Label": 0.1, "water_level": 0.8, "Bottle":0.6, "bad_label": 0.1}
            confidence = {}

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = r.names[cls_id]
                    if label in thresholds:
                        if conf >= thresholds[label]:
                            status[label] = True
                        confidence[label] = conf

            required_keys = ["Cap", "Label", "water_level", "Bottle"]
            final = "PROPER" if all(status.get(k, False) for k in required_keys) else "DEFECT"

            st.markdown(f"### HASIL: **{final}**")
            st.write("Detail Komponen:", status)

            json_file = save_data(status)
            st.success("Hasil deteksi disimpan.")


def save_data(status_dict):

    # location to store data json
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
        "date_checked": time_checked
    }

    existing_data.append(data)

    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

    return filename
