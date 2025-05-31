import streamlit as st
from PIL import Image
import tempfile
import json
import datetime
import os
import pathlib
import subprocess

# Tentukan path project root\ nBASE_DIR = pathlib.Path(__file__).resolve().parent.parent
PY_YOLO = "venv_bottle/Scripts/python.exe"
SCRIPT   = "ml_scripts/classify_bottle.py"

def run():
    st.header("Klasifikasi Botol PROPER / DEFECT")
    uploaded_file = st.file_uploader("Upload gambar botol", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        return

    # Tampilkan preview
    img = Image.open(uploaded_file)
    new_width = 300
    new_height = int(img.height * new_width / img.width)
    resized_img = img.resize((new_width, new_height))
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(resized_img, caption="Gambar yang diunggah")

    # Simpan sementara
    with col2, tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        image_path = tmp.name

    # Payload thresholds
    thresholds = {"Cap": 0.01, "Label": 0.1, "water_level": 0.8, "Bottle":0.6, "bad_label": 0.1}
    payload = {"image_path": image_path, "thresholds": thresholds}

    # Panggil skrip klasifikasi di venv_yolo
    proc = subprocess.Popen(
        [str(PY_YOLO), str(SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = proc.communicate(json.dumps(payload).encode())

    # Ambil JSON baris terakhir dari stdout
    raw = out.decode().strip().splitlines()
    json_line = next((l for l in reversed(raw) if l.startswith("{") and l.endswith("}")), None)
    if not json_line:
        st.error("Gagal mendapatkan hasil classifier:\n" + err.decode().strip())
        return

    status = json.loads(json_line)

    # Tampilkan
    required_keys = ["Cap", "Label", "water_level", "Bottle"]
    final = "PROPER" if all(status.get(k, False) for k in required_keys) else "DEFECT"
    st.markdown(f"### HASIL: **{final}**")
    st.write("Detail Komponen:", {k: status[k] for k in status if k != "final"})

    # Simpan histori
    json_file = save_data(status)
    st.success("Hasil deteksi disimpan: " + json_file)


def save_data(status_dict):
    folder_path = "database_json"
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, "hasil_deteksi_list.json")
    try:
        with open(filename, "r") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    new_id = max((item.get("id", 0) for item in existing), default=0) + 1
    status_dict["id"] = new_id
    status_dict["date_checked"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing.append(status_dict)

    with open(filename, "w") as f:
        json.dump(existing, f, indent=4)
    return filename
