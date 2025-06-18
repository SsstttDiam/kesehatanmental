import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("mentalhealth_model.pkl")
scaler = joblib.load("scaler.pkl")

# Judul halaman
st.set_page_config(page_title="Prediksi Depresi Mahasiswa", layout="centered")
st.title("📊 Prediksi Risiko Depresi Mahasiswa")
st.write("Isi data di bawah ini untuk memprediksi apakah mahasiswa berpotensi mengalami depresi.")

# Form input data
with st.form("mental_health_form"):
    st.subheader("📝 Masukkan Data Mahasiswa:")
    
    stress = st.slider("1. Tingkat Stres", 1, 5, 3, help="1 = sangat rendah, 5 = sangat tinggi")
    sleep = st.slider("2. Kualitas Tidur", 1, 5, 3, help="1 = sangat buruk, 5 = sangat baik")
    academic = st.slider("3. Tekanan Akademik", 1, 5, 3, help="1 = sangat ringan, 5 = sangat berat")
    social = st.slider("4. Dukungan Sosial", 1, 5, 3, help="1 = tidak ada dukungan, 5 = sangat kuat")
    phone = st.number_input("5. Durasi Penggunaan HP per Hari (jam)", min_value=0.0, max_value=24.0, value=5.0, step=0.1)

    # Tombol submit
    submitted = st.form_submit_button("🔍 Prediksi")

# Ketika tombol diklik
if submitted:
    # Proses input
    features = np.array([[stress, sleep, academic, social, phone]])
    features_scaled = scaler.transform(features)
    
    # Prediksi dengan model
    prediction = model.predict(features_scaled)[0]

    # Output hasil
    st.subheader("📢 Hasil Prediksi")
    if prediction == 1:
        st.error("⚠️ Mahasiswa **berpotensi mengalami depresi**.\nSegera beri perhatian dan dukungan yang tepat.")
    else:
        st.success("✅ Mahasiswa **tidak berpotensi mengalami depresi**.\nTetap jaga kesehatan mental dan dukungan sosial.")
