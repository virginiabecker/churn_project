import pandas as pd

def load_data(file_path):
    """Carrega os dados de um arquivo CSV.

    Args:
        file_path (str): O caminho para o arquivo CSV.

    Returns:
        pandas.DataFrame: O DataFrame contendo os dados.
    """
    df = pd.read_csv(file_path)
    return df

if __name__ == '__main__':
    data_path = 'data/churn.csv'
    df = load_data(data_path)
    print("Dados carregados com sucesso!")
    print(df.head())