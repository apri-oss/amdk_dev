import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Load artifacts once with custom_objects for 'mse'
@st.cache_resource
def load_artifacts():
    model = load_model(
        "ml_models/forecasting/model_global_region.h5",
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )
    scaler_temporal = joblib.load("ml_models/forecasting/scaler_temporal.joblib")
    scaler_per_region = joblib.load("ml_models/forecasting/scaler_per_region.joblib")
    label_encoder = joblib.load("ml_models/forecasting/label_encoder.joblib")
    return model, scaler_temporal, scaler_per_region, label_encoder

model, scaler_temporal, scaler_per_region, label_encoder = load_artifacts()


def run():
    st.header("Module 3: Prediksi Time Series")
    st.write("Halaman untuk prediksi permintaan air minum berdasarkan model LSTM kategori Water.")

    # Pilih region
    regions = list(label_encoder.classes_)
    selected_region = st.selectbox("Pilih Region", regions)
    region_code = label_encoder.transform([selected_region])[0]

    # Pilih horizon prediksi
    steps_ahead = st.slider("Jumlah hari prediksi ke depan", min_value=1, max_value=60, value=30)

    # Prediksi
    if st.button("Jalankan Prediksi"):
        today = datetime.now().date()
        look_back = 30
        # Buat DataFrame tanggal historis
        dates_hist = [today - timedelta(days=i) for i in range(look_back)][::-1]
        df_feat = pd.DataFrame({'Order_Date': dates_hist})
        # Convert to datetime64 for .dt access
        df_feat['Order_Date'] = pd.to_datetime(df_feat['Order_Date'])
        # Ekstrak fitur temporal
        df_feat['day_of_year'] = df_feat['Order_Date'].dt.dayofyear
        df_feat['month'] = df_feat['Order_Date'].dt.month
        df_feat['year'] = df_feat['Order_Date'].dt.year

        # Standard scale fitur temporal
        tmp = scaler_temporal.transform(df_feat[['day_of_year','month','year']])
        df_feat[['day_of_year','month','year']] = tmp

        # Buat input sequence (1, look_back, num_regions) dengan kuantitas dummy = 0
        num_regions = len(regions)
        seq = np.zeros((1, look_back, num_regions))
        seq[:, :, region_code] = 0.0

        # Prediksi dan inverse transform
        preds_scaled = model.predict([seq, np.array([region_code])])[0]
        preds = scaler_per_region[selected_region].inverse_transform(preds_scaled.reshape(-1,1)).flatten()

        # Siapkan DataFrame hasil
        dates_pred = [today + timedelta(days=i+1) for i in range(steps_ahead)]
        df_result = pd.DataFrame({
            'Date': dates_pred,
            'Predicted_Quantity': preds[:steps_ahead]
        })

        # Tampilkan grafik dan tabel
        st.subheader("Hasil Prediksi Quantity per Tanggal")
        st.line_chart(df_result.set_index('Date'))
        st.dataframe(df_result)
