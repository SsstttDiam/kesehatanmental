import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Muat model dan scaler
model = joblib.load('mental_health_model.pkl')
scaler = joblib.load('mental_health_scaler.pkl')

st.title("Prediksi Risiko Depresi pada Mahasiswa")

# Tampilkan cuplikan dataset
st.subheader("Contoh Dataset")
df = pd.read_csv("mentalhealth_dataset.csv", sep=';')
df = df[["Gender", "Age", "CGPA", "SleepQuality", "StudyStressLevel",
         "StudyHoursPerWeek", "AcademicEngagement", "HasMentalHealthSupport", "Depression"]]
st.dataframe(df.head(10))

# Form input
with st.form("Form_depresi"):
    st.header("Masukkan data mahasiswa:")
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
    age = st.number_input("Usia", 15, 60, value=20)
    cgpa = st.slider("IPK (CGPA)", 0.0, 4.0, 3.0)
    sleep_quality = st.slider("Kualitas Tidur", 1, 5, 3)
    stress_level = st.slider("Tingkat Stres Studi", 1, 5, 3)
    study_hours = st.slider("Jam Belajar per Minggu", 0, 50, 15)
    engagement = st.slider("Keterlibatan Akademik", 1, 5, 3)
    support = st.selectbox("Punya Dukungan Mental Health?", ["Tidak", "Ya"])

    submit = st.form_submit_button("Proses")

# Prediksi
if submit:
    gender_encoded = 1 if gender == "Male" else 0
    support_encoded = 1 if support == "Ya" else 0

    features = np.array([[gender_encoded, age, cgpa, sleep_quality, stress_level,
                          study_hours, engagement, support_encoded]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    st.header("Hasil Prediksi")
    if prediction == 1:
        st.error("Hasil: Anda berisiko mengalami depresi.")
    else:
        st.success("Hasil: Anda tidak berisiko mengalami depresi.")
