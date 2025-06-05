import streamlit as st
import numpy as np
import joblib

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsão de Diabetes", page_icon="🧠", layout="centered")

# --- TÍTULO ---
st.title("🧠 Previsão de Diabetes com Machine Learning")
st.write("Preencha os dados clínicos abaixo para prever a probabilidade de diabetes.")

# --- CARREGAR MODELO ---
model = joblib.load("src/model.pkl")
scaler = joblib.load("src/scaler.pkl")

# --- ENTRADAS DO USUÁRIO ---
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Número de gestações", min_value=0, value=1)
    glucose = st.number_input("Glicose", min_value=0, value=120)
    blood_pressure = st.number_input("Pressão Arterial", min_value=0, value=70)
    skin_thickness = st.number_input("Espessura da Pele", min_value=0, value=20)

with col2:
    insulin = st.number_input("Nível de Insulina", min_value=0, value=85)
    bmi = st.number_input("IMC (BMI)", min_value=0.0, value=28.0)
    diabetes_pedigree = st.number_input("Histórico Familiar", min_value=0.0, value=0.4)
    age = st.number_input("Idade", min_value=0, value=30)

# --- BOTÃO DE PREVISÃO ---
if st.button("🔍 Prever"):

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    
    # Normalização igual à do treino
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # --- EXIBIÇÃO DO RESULTADO ---
    if prediction == 1:
        st.error(f"🩺 Resultado: Alto risco de diabetes ({prob:.2%} de chance)")
    else:
        st.success(f"✅ Resultado: Baixo risco de diabetes ({prob:.2%} de chance)")

    st.markdown("---")
    st.markdown("🔁 Você pode alterar os valores e clicar novamente para novas previsões.")
