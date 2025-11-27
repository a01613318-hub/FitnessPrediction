import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.write("# Predicción: ¿Estás en forma?")
st.image("fitness.jpg", caption="Predice si estás en forma.")

st.header("Datos del usuario")
# ----------------------------------------------------
# 1. IMPORTAR DATOS
# ----------------------------------------------------
df = pd.read_csv("Fitness_Classification.csv")

# Revisamos nombres de columnas
# st.write(df.columns)

# ----------------------------------------------------
# 2. LIMPIEZA Y CONVERSIÓN DE VARIABLES
# ----------------------------------------------------

# Asegurar que smokes está estandarizado
df["smokes"] = df["smokes"].replace({"no": 0, "yes": 1, "0": 0, "1": 1})

# Asegurar que gender está estandarizado
df["gender"] = df["gender"].replace({"F": 0, "M": 1})

# Convertir sleep_hours a numérico y rellenar NA
df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].mean())

# ----------------------------------------------------
# 3. DEFINIR FEATURES Y TARGET
# ----------------------------------------------------
X = df.drop(["is_fit"], axis=1)
y = df["is_fit"]

# Verificar que no existan NaN ni strings
# st.write(X.isna().sum())

# ----------------------------------------------------
# 4. ENTRENAR MODELO
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, y_train)

# ----------------------------------------------------
# 5. FUNCIÓN PARA CAPTURAR INPUT DEL USUARIO
# ----------------------------------------------------
def user_input_features():

    # NOTA: siempre dar un "value" dentro del rango permitido.
    age = st.number_input("Edad:", min_value=10, max_value=100, value=25)

    height_cm = st.number_input("Altura (cm):", min_value=120, max_value=220, value=170)
    weight_kg = st.number_input("Peso (kg):", min_value=35, max_value=200, value=70)

    heart_rate = st.number_input("Frecuencia cardiaca:", min_value=40.0, max_value=120.0, value=70.0)
    blood_pressure = st.number_input("Presión arterial:", min_value=80.0, max_value=200.0, value=120.0)

    sleep_hours = st.number_input("Horas de sueño:", min_value=3.0, max_value=12.0, value=7.0)
    nutrition_quality = st.number_input("Calidad de nutrición (1–10):", min_value=1.0, max_value=10.0, value=7.0)
    activity_index = st.number_input("Índice de actividad (1–10):", min_value=1.0, max_value=10.0, value=5.0)

    fuma = st.selectbox("¿Fuma?", ("No", "Sí"))
    fuma = 1 if fuma == "Sí" else 0

    gender = st.selectbox("Género:", ("F", "M"))
    gender = 0 if gender == "F" else 1

    # Crear dataframe con los datos del usuario
    datos_usuario = pd.DataFrame({
        "age": [age],
        "height_cm": [height_cm],
        "weight_kg": [weight_kg],
        "heart_rate": [heart_rate],
        "blood_pressure": [blood_pressure],
        "sleep_hours": [sleep_hours],
        "nutrition_quality": [nutrition_quality],
        "activity_index": [activity_index],
        "smokes": [fuma],
        "gender": [gender]
    })

    return datos_usuario

# ----------------------------------------------------
# 6. OBTENER DATOS DEL USUARIO
# ----------------------------------------------------
datos = user_input_features()

# ----------------------------------------------------
# 7. PREDECIR
# ----------------------------------------------------
prediccion = LR.predict(datos)
prob = LR.predict_proba(datos)[0][1]

st.subheader("Resultado")
st.write("Predicción (1 = Fit, 0 = No Fit):", int(prediccion[0]))

st.write("Probabilidad de estar en buena condición física:", round(prob * 100, 2), "%")


