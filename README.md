# 🧠 Previsão de Diabetes com Machine Learning

Este projeto aplica algoritmos de Machine Learning para prever a ocorrência de diabetes com base em dados clínicos. O modelo foi desenvolvido com Python, utilizando `pandas`, `scikit-learn`, `seaborn` e `matplotlib`.

## 📊 Dataset

- Fonte: [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Atributos: glicose, pressão arterial, espessura da pele, insulina, IMC, idade, etc.
- Variável alvo: `Outcome` (0 = não diabético, 1 = diabético)

## 🎯 Objetivos

- Análise exploratória dos dados
- Tratamento e normalização
- Treinamento de modelos (Logistic Regression, Random Forest, KNN)
- Avaliação com métricas e matriz de confusão
- Salvamento do melhor modelo (`.pkl`)

## 📈 Resultados

| Modelo               | Acurácia |
|----------------------|----------|
| Logistic Regression  | 0.740    |
| KNN                  | 0.718    |
| Random Forest        | 0.753 ✅ |

✅ O melhor modelo foi salvo como `model.pkl`.

## 📂 Estrutura

diabetes-predictor/
├── data/ # Dataset CSV
├── notebooks/ # Análise e treino
├── src/ # Funções reutilizáveis
├── requirements.txt
└── README.md


## 🚀 Como Executar

```bash
git clone https://github.com/mateusrodc/diabetes-predictor.git
cd diabetes-predictor
pip install -r requirements.txt
jupyter notebook

🛠️ Tecnologias
Python 3

Pandas, NumPy

Scikit-learn

Seaborn, Matplotlib

Jupyter Notebook

Joblib

🤝 Autor
Mateus Rodrigues
📧 mateusrdcs@gmail.com
🔗 linkedin.com/in/mateusrodc
💻 github.com/mateusrodc


---
