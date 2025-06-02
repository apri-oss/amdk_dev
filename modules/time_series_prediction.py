import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import json

def scaler_to_dict(scaler):
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_samples_seen": int(scaler.n_samples_seen_) if hasattr(scaler, 'n_samples_seen_') else None
    }

def label_encoder_to_list(label_encoder):
    return label_encoder.classes_.tolist()

@st.cache_resource
def load_artifacts():
    model = load_model(
        "ml_models/forecasting/model_global_region.h5",
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )
    scaler_temporal = joblib.load("ml_models/forecasting/scaler_temporal.joblib")
    scaler_per_region = joblib.load("ml_models/forecasting/scaler_per_region.joblib")
    label_encoder = joblib.load("ml_models/forecasting/label_encoder.joblib")

    # Convert scalers to dicts (JSON serializable)
    scaler_temporal_dict = scaler_to_dict(scaler_temporal)

    # scaler_per_region biasanya dictionary {region: scaler}, convert tiap scaler jadi dict juga
    scaler_per_region_dict = {
        region: scaler_to_dict(scaler_per_region[region]) for region in scaler_per_region
    }

    label_encoder_list = label_encoder_to_list(label_encoder)

    return model, scaler_temporal_dict, scaler_per_region_dict, label_encoder_list

model, scaler_temporal, scaler_per_region, label_encoder = load_artifacts()


# Fungsi load scaler dari dict ke StandardScaler di runtime deploy
from sklearn.preprocessing import StandardScaler, LabelEncoder

def dict_to_scaler(d):
    scaler = StandardScaler()
    scaler.mean_ = np.array(d['mean'])
    scaler.scale_ = np.array(d['scale'])
    scaler.var_ = np.array(d['var'])
    if d['n_samples_seen'] is not None:
        scaler.n_samples_seen_ = d['n_samples_seen']
    return scaler

def list_to_label_encoder(classes):
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    return le


# Saat runtime deploy, convert dulu ke object sklearn
scaler_temporal_obj = dict_to_scaler(scaler_temporal)
scaler_per_region_obj = {k: dict_to_scaler(v) for k,v in scaler_per_region.items()}
label_encoder_obj = list_to_label_encoder(label_encoder)


def run():
    st.header("Module 3: Prediksi Time Series")
    st.write("Halaman untuk prediksi permintaan air minum berdasarkan model LSTM kategori Water.")

    regions = list(label_encoder_obj.classes_)
    selected_region = st.selectbox("Pilih Region", regions)
    region_code = label_encoder_obj.transform([selected_region])[0]

    steps_ahead = st.slider("Jumlah hari prediksi ke depan", min_value=1, max_value=60, value=30)

    if st.button("Jalankan Prediksi"):
        today = datetime.now().date()
        look_back = 30
        dates_hist = [today - timedelta(days=i) for i in range(look_back)][::-1]
        df_feat = pd.DataFrame({'Order_Date': dates_hist})
        df_feat['Order_Date'] = pd.to_datetime(df_feat['Order_Date'])
        df_feat['day_of_year'] = df_feat['Order_Date'].dt.dayofyear
        df_feat['month'] = df_feat['Order_Date'].dt.month
        df_feat['year'] = df_feat['Order_Date'].dt.year

        tmp = scaler_temporal_obj.transform(df_feat[['day_of_year','month','year']])
        df_feat[['day_of_year','month','year']] = tmp

        num_regions = len(regions)
        seq = np.zeros((1, look_back, num_regions))
        seq[:, :, region_code] = 0.0

        preds_scaled = model.predict([seq, np.array([region_code])])[0]
        preds = scaler_per_region_obj[selected_region].inverse_transform(preds_scaled.reshape(-1,1)).flatten()

        dates_pred = [today + timedelta(days=i+1) for i in range(steps_ahead)]
        df_result = pd.DataFrame({
            'Date': dates_pred,
            'Predicted_Quantity': preds[:steps_ahead]
        })

        st.subheader("Hasil Prediksi Quantity per Tanggal")
        st.line_chart(df_result.set_index('Date'))
        st.dataframe(df_result)
