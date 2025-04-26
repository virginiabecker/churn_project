import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """Realiza o pré-processamento dos dados."""
    # Converter 'TotalCharges' para numérico, tratando erros
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Converter colunas binárias para 1 e 0 (incluindo 'Churn')
    for col in ['Partner', 'Dependents', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'PaperlessBilling', 'PhoneService', 'Churn']:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Tratar 'No internet service' e 'No phone service' para 'No'
    column_services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in column_services:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No'})
    if 'MultipleLines' in df.columns:
        df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
    if 'PhoneService' in df.columns:
        df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})

    # Calcular 'QtdServicos'
    column_services_qtd = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    services = df[column_services_qtd].applymap(lambda x: 1 if x == 1 else 0) # Assumindo que já são 0/1
    df['QtdServicos'] = services.sum(axis=1)

    # Remover coluna 'customerID' (movido para antes do get_dummies)
    df.drop(['customerID'], axis=1, inplace=True)

    # Converter outras colunas categóricas usando get_dummies
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod', 'gender'] # Adicione 'gender' aqui se ainda não estiver sendo tratado
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # # Converter 'gender' para numérico (removido pois agora é tratado por get_dummies)
    # if 'gender' in df.columns:
    #     df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Separar X e y
    x = df.drop('Churn', axis=1)
    y = df['Churn']

    return df
    
def split_data(x, y, test_size=0.3, random_state=42, stratify=None):
    """Divide os dados em conjuntos de treino e teste."""
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    data_path = 'data/churn.csv'
    df = pd.read_csv(data_path)
    x, y = preprocess_data(df.copy())  # Receba x e y aqui
    X_train, X_test, y_train, y_test = split_data(x, y, stratify=y)
    print("Dados pré-processados e divididos!")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

