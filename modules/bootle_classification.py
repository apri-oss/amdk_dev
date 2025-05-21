import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile
import csv
import datetime
import os

# Inisialisasi Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="5AH3miHLHG3JTapL1Dxy"
)

def simpan_hasil(status_dict, confidence_dict):
    waktu = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_status = "PROPER" if all(status_dict.values()) else "DEFECT"
    with open("hasil_deteksi.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            waktu,
            final_status,
            confidence_dict.get("Cap", 0),
            confidence_dict.get("Label", 0),
            confidence_dict.get("water_level", 0)
        ])

def run():
    st.header("Klasifikasi Botol PROPER / DEFECT")
    uploaded_file = st.file_uploader("Upload gambar botol", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        new_width = 100
        new_height = int(img.height * new_width / img.width)
        resized_img = img.resize((new_width, new_height))
        col1, col2 = st.columns([1, 1])  # proporsi sama

        with col1:
            st.image(resized_img, caption="Gambar yang diunggah")

        with col2:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name)
                result = CLIENT.infer(tmp.name, model_id="vision-lwi3r/1")

            pred = result["predictions"]
            status = {"Cap": False, "Label": False, "water_level": False}
            thresholds = {"Cap": 0.7, "Label": 0.7, "water_level": 0.9}

            confidence = {}
            for p in pred:
                if p["class"] in thresholds:
                    if p["confidence"] >= thresholds[p["class"]]:
                        status[p["class"]] = True
                    confidence[p["class"]] = p["confidence"]

            final = "PROPER" if all(status.values()) else "DEFECT"

            st.markdown(f"### Hasil: **{final}**")
            st.write("Detail Komponen:", status)

            # Simpan hasil ke CSV
            simpan_hasil(status, confidence)
            st.success("Hasil deteksi disimpan.")