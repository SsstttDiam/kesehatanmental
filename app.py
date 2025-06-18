from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load("mentalhealth_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            stress = float(request.form["StressLevel"])
            sleep = float(request.form["SleepQuality"])
            academic = float(request.form["AcademicPressure"])
            social = float(request.form["SocialSupport"])
            phone = float(request.form["PhoneUsageHours"])

            features = np.array([[stress, sleep, academic, social, phone]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            result = "Berpotensi mengalami depresi" if prediction == 1 else "Tidak berpotensi mengalami depresi"
            return render_template("index.html", result=result)
        except:
            return render_template("index.html", result="Masukan tidak valid.")
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
