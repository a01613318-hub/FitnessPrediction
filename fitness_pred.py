import numpy as np
import streamlit as st
import pandas as pd

st.write("# Predicción: ¿Está en forma?")
st.image("fitness.jpg", caption="Predice si una persona está en forma.")

st.header("Datos del usuario")

def user_input_features():

    age = st.number_input("age:", min_value=10, max_value=100, value=10) #tuve que poner estos por que sino marca error
    height_cm = st.number_input("height_cm:", min_value=100, max_value=220, value=100)
    weight_kg = st.number_input("weight_kg:", min_value=30, max_value=200, value=30)
    heart_rate = st.number_input("heart_rate:", min_value=40, max_value=200, value=40)
    blood_pressure = st.number_input("blood_pressure:", min_value=80, max_value=200, value=80)
    sleep_hours = st.number_input("sleep_hours:", min_value=0.0, max_value=15.0, value=1.0)
    nutrition_quality = st.number_input("nutrition_quality:", min_value=0.0, max_value=10.0, value=1.0)
    activity_index = st.number_input("activity_index:", min_value=1.0, max_value=5.0, value=1.0)

    smokes = st.selectbox("smokes:", ["no", "yes"])
    smokes = 1 if smokes == "yes" else 0

    gender = st.selectbox("gender:", ["M", "F"])
    gender = 1 if gender == "M" else 0

    # Diccionario limpio y alineado con el CSV
    user_input_data = {
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "heart_rate": heart_rate,
        "blood_pressure": blood_pressure,
        "sleep_hours": sleep_hours,
        "nutrition_quality": nutrition_quality,
        "activity_index": activity_index,
        "smokes": smokes,
        "gender": gender
    }

    return pd.DataFrame(user_input_data, index=[0])

df = user_input_features()

datos = pd.read_csv("Fitness_Classification.csv", encoding="latin-1")

datos["smokes"] = datos["smokes"].replace({"yes": 1, "no": 0})
datos["gender"] = datos["gender"].replace({"M": 1, "F": 0})

X = datos.drop(columns=["is_fit"])
y = datos["is_fit"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1234
)

LR = LogisticRegression(max_iter=2000)
LR.fit(X_train, y_train)

b1 = LR.coef_[0]
b0 = LR.intercept_[0]

prediccion = (
    b0
    + b1[0] * df["age"]
    + b1[1] * df["height_cm"]
    + b1[2] * df["weight_kg"]
    + b1[3] * df["heart_rate"]
    + b1[4] * df["blood_pressure"]
    + b1[5] * df["sleep_hours"]
    + b1[6] * df["nutrition_quality"]
    + b1[7] * df["activity_index"]
    + b1[8] * df["smokes"]
    + b1[9] * df["gender"]
)

prediccion_prob = 1 / (1 + np.exp(-prediccion))
prediccion_final = (prediccion_prob > 0.5).astype(int)

st.subheader("Predicción final")
st.write("Probabilidad de estar en forma:", float(prediccion_prob))
st.write("¿Está en forma?:", int(prediccion_final))
