import streamlit as st
from streamlit_option_menu import option_menu

# Import module
from modules import bootle_classification, extraction_sentiment, time_series_prediction

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Pilih Module",  # Judul sidebar
        options=["Klasifikasi Botol", "Ekstraksi Sentimen", "Prediksi Time Series"],
        icons=["box", "chat-left-text", "graph-up"],  # Bisa diganti dengan ikon lain
        menu_icon="cast",  # Ikon menu utama
        default_index=0,
    )

# Routing sesuai pilihan
if selected == "Klasifikasi Botol":
    bootle_classification.run()
elif selected == "Ekstraksi Sentimen":
    extraction_sentiment.run()
elif selected == "Prediksi Time Series":
    time_series_prediction.run()
