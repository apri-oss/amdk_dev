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

def prepare_df_botol(data):
    df = pd.DataFrame(data)
    if not df.empty:
        if 'date_checked' in df.columns:
            df['date_checked'] = pd.to_datetime(df['date_checked'])
        else:
            st.warning("Kolom 'date_checked' tidak ditemukan di data botol.")
            return pd.DataFrame()
        required_keys = ["Cap", "Label", "water_level", "Bottle"]
        def is_proper(row):
            return all(row.get(k, False) for k in required_keys)
        df['final_status'] = df.apply(lambda row: "PROPER" if is_proper(row) else "DEFECT", axis=1)
    return df

def prepare_df_sentimen(data):
    df = pd.DataFrame(data)
    return df

def run():
    st.header("Dashboard")
    json_path = "database_json/hasil_deteksi_list.json"
    data_botol = load_data(json_path)
    df_botol = prepare_df_botol(data_botol)

    if df_botol.empty:
        st.warning("Data botol kosong atau file JSON tidak ditemukan.")
    else:
        # ====== Filter Tanggal ======
        tanggal_default = df_botol['date_checked'].dt.date.max()
        tanggal_pilih = st.date_input("Pilih Tanggal Pengecekan", value=tanggal_default)
        filtered_df = df_botol[df_botol['date_checked'].dt.date == tanggal_pilih]

        total_data = len(filtered_df)
        st.markdown(f"**Total data: {total_data}**")

        if total_data == 0:
            st.info("Tidak ada data pada tanggal ini.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribusi Status Botol")
                status_counts = filtered_df['final_status'].value_counts()
                colors = ['#1f77b4', '#d62728']
                labels = status_counts.index.tolist()
                sizes = status_counts.values.tolist()
                fig, ax = plt.subplots()
                fig.patch.set_alpha(0) 
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
                        if f in defect_df.columns:
                            penyebab[f] = (defect_df[f] == False).sum()
                        else:
                            penyebab[f] = 0
                    for f in faktor_true:
                        if f in defect_df.columns:
                            penyebab[f] = (defect_df[f] == True).sum()
                        else:
                            penyebab[f] = 0
                    penyebab_series = pd.Series(penyebab)
                    penyebab_series.index = penyebab_series.index.map(fitur_map)
                    penyebab_series = penyebab_series.sort_values(ascending=False)
                    st.bar_chart(penyebab_series)
                    st.table(penyebab_series.to_frame(name="Jumlah"))
                else:
                    st.info("Tidak ada data DEFECT pada tanggal ini.")

    # ====== Analisis Sentimen ======
    st.markdown("---")
    st.subheader("Analisis Sentimen")
    json_path_sentimen = "database_json/data_sentimen.json"
    data_sentimen = load_data(json_path_sentimen)
    df_sentimen = prepare_df_sentimen(data_sentimen)
    if df_sentimen.empty:
        st.warning("Data sentimen kosong atau file JSON tidak ditemukan.")
        return
    # Filter entitas
    entitas_list = df_sentimen['entitas'].unique().tolist()
    entitas_list.insert(0, "Semua")
    pilih_entitas = st.selectbox("Pilih Entitas", entitas_list)
    # Filter berdasarkan entitas
    if pilih_entitas != "Semua":
        df_sentimen_filtered = df_sentimen[df_sentimen['entitas'] == pilih_entitas]
    else:
        df_sentimen_filtered = df_sentimen
    total_data = len(df_sentimen_filtered)
    st.markdown(f"**Total data sentimen: {total_data}**")
    if total_data == 0:
        st.info("Tidak ada data untuk entitas ini.")
        return

    # ====== Layout Dua Kolom ======
    col1, col2 = st.columns(2)
    # === Kolom Kiri: Pie Chart ===
    with col1:
        st.subheader("Distribusi Sentimen")
        sentimen_count = df_sentimen_filtered['is_sentimen'].value_counts().rename(index={True: 'Positif', False: 'Negatif'})
        colors = ['#1f77b4', '#d62728']
        labels_order = ['Positif', 'Negatif']
        sizes = [sentimen_count.get(label, 0) for label in labels_order]
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels_order,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops=dict(color="w")
        )
        ax.axis('equal')
        plt.setp(autotexts, size=12, weight="bold")
        st.pyplot(fig)

    # === Kolom Kanan: Bar Chart ===
    with col2:
        st.subheader("Distribusi Entitas per Sentimen")
        entitas_summary = df_sentimen_filtered.groupby(['entitas', 'is_sentimen']).size().unstack(fill_value=0)
        # Pastikan kedua kolom sentimen selalu ada
        if True not in entitas_summary.columns:
            entitas_summary[True] = 0
        if False not in entitas_summary.columns:
            entitas_summary[False] = 0
        entitas_summary.rename(columns={True: 'Positif', False: 'Negatif'}, inplace=True)
        entitas_summary = entitas_summary[['Positif', 'Negatif']]
        if not entitas_summary.empty:
            st.bar_chart(entitas_summary)
        else:
            st.info("Tidak ada distribusi entitas tersedia.")
    # === Tabel Detail di Bawah ===
    st.subheader("Detail Data Sentimen")
    df_tabel = df_sentimen_filtered[['entitas', 'kota', 'text', 'is_sentimen']].copy()
    df_tabel['Sentimen'] = df_tabel['is_sentimen'].map({True: 'Positif', False: 'Negatif'})
    df_tabel = df_tabel[['entitas', 'kota', 'text', 'Sentimen']]
    # st.dataframe(df_tabel)
    st.dataframe(df_tabel, width=1200)
