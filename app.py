import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
try:
    model = joblib.load("mentalhealth_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("❌ Model atau scaler tidak ditemukan. Pastikan 'mentalhealth_model.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
    st.stop()

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Depresi Mahasiswa", layout="centered")
st.title("🧠 Prediksi Risiko Depresi Mahasiswa")
st.markdown(
    """
    Aplikasi ini digunakan untuk memprediksi apakah seorang mahasiswa **berpotensi mengalami depresi**
    berdasarkan lima indikator utama:
    - Tingkat stres
    - Kualitas tidur
    - Tekanan akademik
    - Dukungan sosial
    - Durasi penggunaan HP per hari
    """
)

# Form input data pengguna
with st.form("mental_health_form"):
    st.subheader("📋 Masukkan Data Mahasiswa:")
    
    stress = st.slider("1️⃣ Tingkat Stres", 1, 5, 3, help="1 = sangat rendah, 5 = sangat tinggi")
    sleep = st.slider("2️⃣ Kualitas Tidur", 1, 5, 3, help="1 = sangat buruk, 5 = sangat baik")
    academic = st.slider("3️⃣ Tekanan Akademik", 1, 5, 3, help="1 = sangat ringan, 5 = sangat berat")
    social = st.slider("4️⃣ Dukungan Sosial", 1, 5, 3, help="1 = tidak ada dukungan, 5 = sangat tinggi")
    phone = st.number_input("5️⃣ Durasi Penggunaan HP per Hari (jam)", min_value=0.0, max_value=24.0, value=5.0, step=0.1)

    submitted = st.form_submit_button("🔍 Prediksi")

# Proses prediksi jika tombol ditekan
if submitted:
    # Konversi dan normalisasi input
    input_data = np.array([[stress, sleep, academic, social, phone]])
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    # Output hasil prediksi
    st.subheader("📢 Hasil Prediksi:")
    if prediction == 1:
        st.error("⚠️ Mahasiswa **berpotensi mengalami depresi**. Disarankan untuk mendapatkan perhatian dan dukungan yang tepat.")
    else:
        st.success("✅ Mahasiswa **tidak berpotensi mengalami depresi**. Tetap jaga kesehatan mental!")

    # Tampilkan data input untuk referensi
    st.markdown("### 🔎 Data yang Dimasukkan:")
    st.json({
        "Tingkat Stres": stress,
        "Kualitas Tidur": sleep,
        "Tekanan Akademik": academic,
        "Dukungan Sosial": social,
        "Durasi HP (jam/hari)": phone
    })
