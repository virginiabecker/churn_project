import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """Realiza o pré-processamento dos dados."""

    # Remover espaços em branco de todas as strings
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Converter 'TotalCharges' para numérico, tratando erros
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Tratar 'No internet service' para 'No' nos serviços
    column_services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in column_services:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No'})

    # Tratar 'No phone service' para 'No' em 'MultipleLines'
    if 'MultipleLines' in df.columns:
        df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

    # Mapear respostas binárias para 0 e 1
    cols_binarias = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'PaperlessBilling', 'Churn'
    ]
    for col in cols_binarias:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).astype('float64')

    # Criar 'QtdServicos' - quantidade de serviços contratados
    column_services_qtd = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    services = df[column_services_qtd].apply(lambda col: col.map(lambda x: 1 if x == 1 else 0)).astype('float64')
    df['QtdServicos'] = services.sum(axis=1).astype('float64')

    # Remover coluna 'customerID'
    if 'customerID' in df.columns:
        df.drop(['customerID'], axis=1, inplace=True)

    # Converter outras colunas categóricas usando get_dummies
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod', 'gender']
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)

    # Identificar colunas numéricas
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remover colunas numéricas NaN antes de imputar
    valid_numerical_cols = [col for col in numerical_cols if df[col].notna().any()]

    # Imputação dos valores faltantes
    imputer = SimpleImputer(strategy='median')
    df[valid_numerical_cols] = imputer.fit_transform(df[valid_numerical_cols])

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
    df_processed = preprocess_data(df.copy())
    x = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    X_train, X_test, y_train, y_test = split_data(x, y, stratify=y)
    print("Dados pré-processados e divididos!")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
