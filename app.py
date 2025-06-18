import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("mentalhealth_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Risiko Depresi Mahasiswa")

# Form input
with st.form("mental_health_form"):
    st.header("Masukkan data mahasiswa:")
    
    stress = st.slider("Tingkat Stres (1 = rendah, 5 = tinggi)", 1, 5, 3)
    sleep = st.slider("Kualitas Tidur (1 = buruk, 5 = sangat baik)", 1, 5, 3)
    academic = st.slider("Tekanan Akademik (1 = ringan, 5 = berat)", 1, 5, 3)
    social = st.slider("Dukungan Sosial (1 = sangat rendah, 5 = sangat tinggi)", 1, 5, 3)
    phone = st.number_input("Durasi Penggunaan HP per Hari (jam)", min_value=0.0, max_value=24.0, value=5.0, step=0.1)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Persiapan data
    features = np.array([[stress, sleep, academic, social, phone]])
    features_scaled = scaler.transform(features)
    
    # Prediksi
    prediction = model.predict(features_scaled)[0]
    
    st.header("Hasil Prediksi")
    if prediction == 1:
        st.error("⚠️ Berpotensi mengalami depresi")
    else:
        st.success("✅ Tidak berpotensi mengalami depresi")
