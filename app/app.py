import io
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from firebase_conn import get_firestore_client

db = get_firestore_client()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsão de Diabetes", page_icon="🧠", layout="centered")

# --- TÍTULO ---
st.title("🧠 Previsão de Diabetes com Machine Learning")
st.write("Preencha os dados clínicos abaixo para prever a probabilidade de diabetes.")

# --- CARREGAR MODELO ---
model = joblib.load("src/model.pkl")
scaler = joblib.load("src/scaler.pkl")

sexo = st.radio("Sexo", ["Mulher", "Homem"], horizontal=True)

# --- ENTRADAS DO USUÁRIO ---
col1, col2 = st.columns(2)

with col1:
    is_mulher = (sexo == "Mulher")
    pregnancies = st.number_input(
        "Número de gestações",
        min_value=0,
        value=1 if is_mulher else 0,
        disabled=not is_mulher
    )

    glucose = st.number_input("Glicose", min_value=0, value=120)
    if glucose < 50 or glucose > 300:
        st.warning("⚠️ Glicose fora da faixa clínica comum (50–300 mg/dL)")

    blood_pressure = st.number_input("Pressão Arterial", min_value=0, value=70)
    if blood_pressure < 40 or blood_pressure > 200:
        st.warning("⚠️ Pressão arterial fora da faixa clínica comum (40–200 mmHg)")

    skin_thickness = st.number_input("Espessura da Pele", min_value=0, value=20)
    if skin_thickness > 100:
        st.warning("⚠️ Espessura da pele incomum (> 100 mm)")

with col2:
    insulin = st.number_input("Nível de Insulina", min_value=0, value=85)
    if insulin > 900:
        st.warning("⚠️ Nível de insulina fora da faixa comum (0–900 µU/mL)")

    bmi = st.number_input("IMC (BMI)", min_value=0.0, value=28.0)
    if bmi < 10 or bmi > 60:
        st.warning("⚠️ IMC fora da faixa clínica comum (10.0–60.0)")

    diabetes_pedigree = st.number_input("Histórico Familiar", min_value=0.0, value=0.4)
    if diabetes_pedigree > 2.5:
        st.warning("⚠️ Valor alto para histórico familiar (> 2.5)")

    age = st.number_input("Idade", min_value=0, value=30)
    if age < 10 or age > 100:
        st.warning("⚠️ Idade fora da faixa comum (10–100 anos)")

# --- BOTÃO DE PREVISÃO ---
if st.button("🔍 Prever"):

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    
    # Normalização igual à do treino
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # --- SALVAR HISTÓRICO DE PREVISÕES ---
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Sexo": sexo,
        "Gestações": pregnancies,
        "Glicose": glucose,
        "Pressão": blood_pressure,
        "Pele": skin_thickness,
        "Insulina": insulin,
        "IMC": bmi,
        "Hist. Familiar": diabetes_pedigree,
        "Idade": age,
        "Probabilidade (%)": f"{prob:.2%}",
        "Resultado": "Com Diabetes" if prediction == 1 else "Sem Diabetes"
    })

    # --- EXIBIÇÃO DO RESULTADO ---
    if prediction == 1:
        st.error(f"🩺 Resultado: Alto risco de diabetes ({prob:.2%} de chance)")
    else:
        st.success(f"✅ Resultado: Baixo risco de diabetes ({prob:.2%} de chance)")

    st.markdown("---")
    st.markdown("🔁 Você pode alterar os valores e clicar novamente para novas previsões.")


    # --- GRÁFICO DE PROBABILIDADES ---
    st.markdown("### 📊 Probabilidades do Modelo")
    
    classes = ['Sem Diabetes', 'Com Diabetes']
    probs = model.predict_proba(input_scaled)[0]

    fig, ax = plt.subplots()
    ax.barh(classes, probs, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilidade')
    ax.set_title('Distribuição das Probabilidades')
    st.pyplot(fig)

    # --- HISTÓRICO DE PREVISÕES (ESTATÍSTICAS) ---
    st.markdown("### 📊 Estatísticas do Histórico")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)

        total = len(df_history)
        com_diabetes = df_history[df_history["Resultado"] == "Com Diabetes"]
        perc_diabetes = (len(com_diabetes) / total) * 100

        st.markdown(f"- **Total de Previsões:** {total}")
        st.markdown(f"- **Com Diabetes:** {len(com_diabetes)} ({perc_diabetes:.2f}%)")
        st.markdown(f"- **Sem Diabetes:** {total - len(com_diabetes)} ({100 - perc_diabetes:.2f}%)")

        st.markdown("#### 📈 Médias dos Parâmetros Informados:")
        stats = df_history.drop(columns=["Sexo", "Probabilidade (%)", "Resultado"]).astype(float).mean()
        st.dataframe(stats.to_frame(name="Média"), use_container_width=True)

    else:
        st.info("Nenhuma previsão realizada ainda.")

    db.collection("previsoes").add({
    "sexo": sexo,
    "idade": age,
    "imc": bmi,
    "probabilidade": float(prob),
    "resultado": "Com Diabetes" if prediction == 1 else "Sem Diabetes"
})
