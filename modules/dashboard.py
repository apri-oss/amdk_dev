import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                return []
    return []

def prepare_df(data):
    df = pd.DataFrame(data)
    if not df.empty:
        df['date_checked'] = pd.to_datetime(df['date_checked'])
        required_keys = ["Cap", "Label", "water_level", "Bottle"]
        df['final_status'] = df.apply(lambda row: "PROPER" if all(row[k] for k in required_keys) else "DEFECT", axis=1)
    return df

def run():
    st.header("Dashboard")
    json_path = "database_json/hasil_deteksi_list.json"
    data = load_data(json_path)
    df = prepare_df(data)

    if df.empty:
        st.warning("Data kosong atau file JSON tidak ditemukan.")
        return
    
    # buat piechart untuk visualisasi
    total_data = len(df)
    st.subheader("Distribusi Status Botol")
    st.subheader(f"**Total data:** {total_data}")

    status_counts = df['final_status'].value_counts()
    colors = ['#1f77b4', '#d62728']  # biru untuk PROPER, merah untuk DEFECT
    labels = status_counts.index.tolist()
    sizes = status_counts.values.tolist()
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)  # bikin background transparan
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
        textprops=dict(color="w")
    )
    ax.axis('equal')  # supaya lingkaran bulat
    plt.setp(autotexts, size=12, weight="bold")
    ax.set_title("Distribusi Status Botol")

    st.pyplot(fig)
    
    # Analisis faktor penyebab botol DEFECT
    fitur_map = {
        "Cap": "Tutup Botol",
        "Label": "Label Merk Hilang",
        "water_level": "Volume Air",
        "Bottle": "Kondisi Botol",
        "bad_label": "Label Merk Rusak"
    }

    fitur = list(fitur_map.keys())

    st.subheader("Faktor yang Mempengaruhi Botol DEFECT")
    defect_df = df[df['final_status'] == "DEFECT"]
    frekuensi_defect = defect_df[fitur].sum().sort_values(ascending=False)
    frekuensi_defect.index = frekuensi_defect.index.map(fitur_map)
    st.bar_chart(frekuensi_defect)
    st.table(frekuensi_defect.to_frame(name="Jumlah"))

 


 
