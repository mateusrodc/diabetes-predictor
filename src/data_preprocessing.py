import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Substitui zeros por NaN onde não fazem sentido
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

    # Preenchimento com a mediana
    df.fillna(df.median(), inplace=True)
    return df

def split_features_labels(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

def normalize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
