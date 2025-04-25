import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """Realiza o pré-processamento dos dados."""
    # Converter 'Churn' para numérico
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Tratar 'TotalCharges'
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan).astype(float)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Criar coluna de quantidade de serviços
    column_services = [
        'PhoneService',
        'MultipleLines',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]
    services = df[column_services].replace({'No internet service': 'No', 'No phone service': 'No'})
    df['QtdServicos'] = services.apply(lambda linha: sum(linha == 'Yes'), axis=1)

    # Remover coluna 'customerID'
    df.drop(['customerID'], axis=1, inplace=True)

    # Aplicar get_dummies
    df = pd.get_dummies(df, drop_first=True)

    # Separar X e y
    x = df.drop('Churn', axis=1)
    y = df['Churn']

    return x, y

def split_data(X, y, test_size=0.3, random_state=42, stratify=None):
    """Divide os dados em conjuntos de treino e teste."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Exemplo de uso
    data_path = '../data/churn.csv'
    df = pd.read_csv(data_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, stratify=y)
    print("Dados pré-processados e divididos!")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)