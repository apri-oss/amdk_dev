import streamlit as st
import pandas as pd
import numpy as np
import json
import os

import google.generativeai as genai

genai.configure(api_key="AIzaSyAisH1Eb0LxI9ZMPlBQ-vU5dmICJIz3c3g")

model = genai.GenerativeModel("gemini-2.0-flash")

def SentimenNer(text):
  prompt = """
  Buatlah representasi JSON untuk analisis sentimen dari sebuah kalimat.
  Setiap objek JSON harus memiliki kunci "entitas", "kota", "is_sentimen", dan "text".

  Aturannya adalah:

  Jika kalimat menyebutkan "air", "botol", atau "label", nilai "entitas" adalah kata tersebut.
  Jika ada menyebutkan rasa berarti masuk "air", bentuk ke "botol", tulisan ke "label"
  Jika kalimat menyebutkan nama kota, nilai "kota" adalah nama kota tersebut. Jika tidak ada, isi dengan "null".
  Jika komentar terkait "air", "botol", atau "label" bersifat positif, nilai "sentimen" adalah true. Jika tidak terkait atau negatif, nilai "sentimen" adalah false.
  Jika entitas dalam kalimat tidak terkait dengan "air", "botol", atau "label", nilai "entitas" adalah "lain-lain".
  Nilai "text" adalah kalimat aslinya.
  Contoh input: "air itu asam, dan botolnya penyok"
  Contoh output:
  { "entitas":"air", "kota":"null","is_sentimen":false, "text":"air itu asam, dan botolnya penyok"}
  tampilkan hanya json nya saja

  Text:" """+text+""" "
  """
  response = model.generate_content(prompt)
  return response.text

def tambah_data_json(file_path, new_data):
    """
    Fungsi untuk menambahkan data ke file JSON dengan ID otomatis.
    Jika file tidak ada, akan dibuat baru dengan ID mulai dari 1.
    Jika sudah ada, ID akan diincrement dari ID terakhir.

    Parameters:
        file_path (str): Path/lokasi file JSON
        new_data (dict): Data baru yang akan ditambahkan (tanpa ID)
    """

    data = []
    last_id = 0  # Default ID jika tidak ada data

    # Cek apakah file sudah ada
    if os.path.exists(file_path):
        try:
            # Buka file dan baca data yang ada
            with open(file_path, 'r') as file:
                data = json.load(file)

                # Cari ID terakhir jika data ada
                if data:
                    if isinstance(data, list):
                        last_id = max(item.get('id', 0) for item in data)
                    else:
                        # Jika data bukan list (single object)
                        last_id = data.get('id', 0)
        except json.JSONDecodeError:
            # Jika file kosong atau corrupt, mulai dengan list kosong
            data = []

    # Generate ID baru
    new_id = last_id + 1
    new_data_with_id = {'id': new_id, **new_data}

    # Tambahkan data baru
    if isinstance(data, list):
        data.append(new_data_with_id)
    else:
        # Jika data bukan list (misalnya dict), ubah ke list dulu
        data = [data, new_data_with_id]

    # No need to create directory for a file in the current path

    # Tulis kembali ke file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Ganti fungsi gabungkan_kolom dengan:
def gabungkan_kolom(df):
    return df.apply(lambda row: ', '.join([f"{val}" for col, val in row.items()]), axis=1)

def hasilEktrasksi(text):
    input = SentimenNer(text)
    lines = input.strip().splitlines()
    cleaned_json_string = ""
    if lines[0].strip() == "```json" and lines[-1].strip() == "```":
        cleaned_json_string = "\n".join(lines[1:-1])
    elif lines[0].strip().startswith("```json") and input.text.strip().endswith("```"): # Kasus jika ```json menempel dengan {
        temp_text = input.text.strip()
        temp_text = temp_text[len("```json"):] # Hapus ```json dari awal
        temp_text = temp_text[:-len("```")] # Hapus ``` dari akhir
        cleaned_json_string = temp_text.strip() # Bersihkan spasi ekstra
    # Path file JSON
    file_json = "database_json/data_sentimen.json"
    data_dict = json.loads(cleaned_json_string)    
    # Panggil fungsi untuk menambahkan data
    tambah_data_json(file_json, data_dict)
    return data_dict
    
def run():
    st.header("Module 2: Ekstraksi Sentimen")
    st.write("Ini adalah halaman untuk ekstraksi sentimen dari teks.")
    # Tambahkan kode ekstraksi sentimen di sini

    # Upload file
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # Baca file CSV
            df = pd.read_csv(uploaded_file)
            df = gabungkan_kolom(df)
            # Simpan dataframe di session state
            st.session_state['df'] = df
            
            # Tampilkan preview data
            st.subheader("Preview Data")
            st.dataframe(df.head())
            
            # Tampilkan jumlah kolom dan baris
            st.write(f"Data memiliki {df.shape[0]} baris")
            data_ektraksi=[]
            for col, val in df.items():
                hasil = hasilEktrasksi(val)
                data_ektraksi.append(hasil)
            # Hasil Ekstraksi
            data_ektraksi = pd.DataFrame(data_ektraksi)
            data_ektraksi['is_sentimen'] = data_ektraksi['is_sentimen'].replace({
                True: 'positif',
                False: 'negatif'
            })
            st.subheader("Hasil Ekstraksi Sentimen Pelanggan")
            st.dataframe(data_ektraksi.head())
        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
    else:
        st.info("Silakan upload file CSV untuk memulai")