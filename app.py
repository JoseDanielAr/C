import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo y el escalador
scaler = joblib.load("escalador.bin")
model = joblib.load("modelo_knn.bin")

# Configuración de la aplicación
st.title("Asistente AI para Cardiólogos")
st.write("Esta aplicación utiliza inteligencia artificial para predecir posibles problemas cardíacos basados en la edad y los niveles de colesterol del paciente.")

# Tab para ingresar datos
tab1 = st.tabs(["Capturar Datos"])[0]

with tab1:
    st.header("Ingrese los Datos del Paciente")
    st.write("Por favor, ingrese la edad y el nivel de colesterol para evaluar el riesgo de problemas cardíacos.")
    
    edad = st.number_input("Edad", min_value=18, max_value=80, value=30, step=1)
    colesterol = st.number_input("Colesterol", min_value=50, max_value=600, value=200, step=1)
    
    if st.button("Predecir Problema Cardíaco"):
        datos = pd.DataFrame([[edad, colesterol]], columns=["Edad", "Colesterol"])
        datos_escalados = scaler.transform(datos)
        prediccion = model.predict(datos_escalados)
        
        st.header("Resultado de la Predicción")
        if prediccion[0] == 1:
            st.error("El paciente podría tener un problema cardíaco.")
            st.image("https://www.clinicadeloccidente.com/wp-content/uploads/2022/04/Blog_1_1-scaled.jpg", caption="Síntomas de Problemas del Corazón")
        else:
            st.success("El paciente no tiene problemas cardíacos según la predicción del modelo.")
