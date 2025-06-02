import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
from datetime import datetime, timedelta

# Fungsi load scaler dari file JSON
def load_scaler_json(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(params['mean'])
    scaler.scale_ = np.array(params['scale'])
    scaler.var_ = np.array(params['var'])
    if params['n_samples_seen'] is not None:
        scaler.n_samples_seen_ = params['n_samples_seen']
    return scaler

# Fungsi load label encoder dari file JSON
def load_label_encoder_json(filename):
    with open(filename, 'r') as f:
        classes = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    return le

# Cache untuk load model dan scaler hanya sekali
@st.cache_resource
def load_artifacts():
    model = load_model(
        "ml_models/forecasting/model_global_region.h5",
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )
    scaler_temporal = load_scaler_json("ml_models/forecasting/scaler_temporal.json")
    scaler_per_region = {}
    with open("ml_models/forecasting/scaler_per_region.json", 'r') as f:
        scaler_per_region_dict = json.load(f)
    # Convert each scaler per region dict ke StandardScaler object
    for region, scaler_params in scaler_per_region_dict.items():
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_params['mean'])
        scaler.scale_ = np.array(scaler_params['scale'])
        scaler.var_ = np.array(scaler_params['var'])
        if scaler_params['n_samples_seen'] is not None:
            scaler.n_samples_seen_ = scaler_params['n_samples_seen']
        scaler_per_region[region] = scaler

    label_encoder = load_label_encoder_json("ml_models/forecasting/label_encoder_classes.json")

    return model, scaler_temporal, scaler_per_region, label_encoder

model, scaler_temporal, scaler_per_region, label_encoder = load_artifacts()

def run():
    st.header("Module 3: Prediksi Time Series")
    st.write("Halaman untuk prediksi permintaan air minum berdasarkan model LSTM kategori Water.")

    regions = list(label_encoder.classes_)
    selected_region = st.selectbox("Pilih Region", regions)
    region_code = label_encoder.transform([selected_region])[0]

    steps_ahead = st.slider("Jumlah hari prediksi ke depan", min_value=1, max_value=60, value=30)

    if st.button("Jalankan Prediksi"):
        today = datetime.now().date()
        look_back = 30

        # DataFrame tanggal historis (fitur temporal)
        dates_hist = [today - timedelta(days=i) for i in range(look_back)][::-1]
        df_feat = pd.DataFrame({'Order_Date': dates_hist})
        df_feat['Order_Date'] = pd.to_datetime(df_feat['Order_Date'])
        df_feat['day_of_year'] = df_feat['Order_Date'].dt.dayofyear
        df_feat['month'] = df_feat['Order_Date'].dt.month
        df_feat['year'] = df_feat['Order_Date'].dt.year

        # Scaling fitur temporal
        tmp = scaler_temporal.transform(df_feat[['day_of_year', 'month', 'year']])
        df_feat[['day_of_year', 'month', 'year']] = tmp

        num_regions = len(regions)
        seq = np.zeros((1, look_back, num_regions))
        seq[:, :, region_code] = 0.0  # kuantitas dummy sesuai kode region

        # Prediksi scaled output
        preds_scaled = model.predict([seq, np.array([region_code])])[0]

        # Inverse scaling hasil prediksi untuk region terpilih
        preds = scaler_per_region[selected_region].inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        dates_pred = [today + timedelta(days=i + 1) for i in range(steps_ahead)]
        df_result = pd.DataFrame({
            'Date': dates_pred,
            'Predicted_Quantity': preds[:steps_ahead]
        })

        st.subheader("Hasil Prediksi Quantity per Tanggal")
        st.line_chart(df_result.set_index('Date'))
        st.dataframe(df_result)

if __name__ == "__main__":
    run()
