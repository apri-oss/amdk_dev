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

    # ====== Filter Tanggal ======
    tanggal_default = max(df['date_checked'].dt.date)
    tanggal_pilih = st.date_input("Pilih Tanggal Pengecekan", value=tanggal_default)
    filtered_df = df[df['date_checked'].dt.date == tanggal_pilih]

    total_data = len(filtered_df)
    st.markdown(f"**Total data: {total_data}**")

    if total_data == 0:
        st.info("Tidak ada data pada tanggal ini.")
        return

    col1, col2 = st.columns(2)

    # ====== Pie Chart (Status Botol) ======
    with col1:
        st.subheader("Distribusi Status Botol")
        status_counts = filtered_df['final_status'].value_counts()
        colors = ['#1f77b4', '#d62728']  # biru: PROPER, merah: DEFECT
        labels = status_counts.index.tolist()
        sizes = status_counts.values.tolist()

        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)  # background transparan
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
            textprops=dict(color="w")
        )
        ax.axis('equal')
        plt.setp(autotexts, size=12, weight="bold")
        st.pyplot(fig)

    # ====== Bar Chart (Faktor DEFECT) ======
    with col2:
        st.subheader("Faktor Penyebab Botol DEFECT")
        fitur_map = {
            "Cap": "Tutup Botol",
            "Label": "Label Merk Hilang",
            "water_level": "Volume Air",
            "Bottle": "Kondisi Botol",
            "bad_label": "Label Merk Rusak"
        }

        defect_df = filtered_df[filtered_df['final_status'] == "DEFECT"]
        if not defect_df.empty:
            faktor_false = ["Cap", "Label", "water_level", "Bottle"]
            faktor_true = ["bad_label"]
            penyebab = {}

            for f in faktor_false:
                penyebab[f] = (defect_df[f] == False).sum()
            for f in faktor_true:
                penyebab[f] = (defect_df[f] == True).sum()

            penyebab_series = pd.Series(penyebab)
            penyebab_series.index = penyebab_series.index.map(fitur_map)
            penyebab_series = penyebab_series.sort_values(ascending=False)

            st.bar_chart(penyebab_series)
            st.table(penyebab_series.to_frame(name="Jumlah"))
        else:
            st.info("Tidak ada data DEFECT pada tanggal ini.")

    # ====== Placeholder Sentimen ======
    st.markdown("---")
    st.subheader("Analisis Sentimen")
    st.info("Visualisasi dari analisis sentimen akan ditampilkan di sini setelah modul sentimen aktif.")
