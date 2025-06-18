import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

np.random.seed(42)
n_samples = 200
df = pd.DataFrame({
    'StressLevel': np.random.randint(1, 6, n_samples),
    'SleepQuality': np.random.randint(1, 6, n_samples),
    'AcademicPressure': np.random.randint(1, 6, n_samples),
    'SocialSupport': np.random.randint(1, 6, n_samples),
    'PhoneUsageHours': np.random.normal(5, 1.5, n_samples)
})

df['Depression'] = (
    (df['StressLevel'] > 3) & 
    (df['AcademicPressure'] > 3) & 
    (df['SocialSupport'] < 3)
).astype(int)

X = df[['StressLevel', 'SleepQuality', 'AcademicPressure', 'SocialSupport', 'PhoneUsageHours']]
y = df['Depression']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Akurasi:", round(acc * 100, 2), "%")
print(classification_report(y_test, y_pred))

joblib.dump(model, "mentalhealth_model.pkl")
joblib.dump(scaler, "scaler.pkl")
