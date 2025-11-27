import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción: ¿Estás en forma? ''')
st.image("fitness.jpg", caption="Predice si una persona está en forma.")

st.header('Datos del usuario')

def user_input_features():

    age = st.number_input("edad:", min_value=10, max_value=100, value=11) #tuve qiue añadir 1+ a l valor mínimo por que streamlit  me marcaba error
    height_cm = st.number_input("altura_cm:", min_value=100, max_value=220, value=101)
    weight_kg = st.number_input("peso_kg:", min_value=30, max_value=200, value=31)
    heart_rate = st.number_input("frecuencia_cardiaca:", min_value=40, max_value=200, value=41)
    blood_pressure = st.number_input("presión_arterial:", min_value=80, max_value=200, value=81)
    sleep_hours = st.number_input("horas_sueño:", min_value=0.0, max_value=15.0, value=0.1)
    nutrition_quality = st.number_input("calidad_nutricional:", min_value=0.0, max_value=10.0, value=0.1)
    activity_index = st.number_input("índice_actividad:", min_value=1.0, max_value=5.0, value=1.1)

    smokes = st.selectbox("fuma:", ["no", "sí"])
    smokes = 1 if smokes == "sí" else 0

    gender = st.selectbox("género:", ["M", "F"])
    gender = 1 if gender == "M" else 0   # codificación simple

    user_input_data = {
        "edad": age,
        "altura_cm": height_cm,
        "peso_kg": weight_kg,
        "frecuencia_cardiaca": heart_rate,
        "presión_arterial": blood_pressure,
        "horas_sueño": sleep_hours,
        "calidad_nutricional": nutrition_quality,
        "índice_actividad": activity_index,
        "fuma": smokes,
        "género": gender
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

datos = pd.read_csv('Fitness_Classification.csv', encoding='latin-1')

datos["fuma"] = datos["fuma"].replace({"sí":1, "no":0})
datos["género"] = datos["género"].replace({"M":1, "F":0})

X = datos.drop(columns=["está_en_forma"])
y = datos["está_en_forma"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1234)

LR = LogisticRegression(max_iter=2000)
LR.fit(X_train, y_train)

b1 = LR.coef_[0]
b0 = LR.intercept_[0]

prediccion = (
    b0
    + b1[0] * df["edad"] #aquí tuve que ponerlos así por que sino streamlit marca error
    + b1[1] * df["altura_cm"]
    + b1[2] * df["peso_kg"]
    + b1[3] * df["frecuencia_cardiaca"]
    + b1[4] * df["presión_arterial"]
    + b1[5] * df["horas_sueño"]
    + b1[6] * df["calidad_nutricional"]
    + b1[7] * df["índice_actividad"]
    + b1[8] * df["fuma"]
    + b1[9] * df["género"]
)


prediccion_prob = 1 / (1 + np.exp(-prediccion))
prediccion_final = (prediccion_prob > 0.5).astype(int)

st.subheader("Predicción final")
st.write("Probabilidad de estar en forma:", float(prediccion_prob))
st.write("¿Está en forma?:", int(prediccion_final))
