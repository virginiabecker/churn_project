import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

def plot_distribution(df, column, title, bins=20):
    """Gera um histograma."""
    df.hist(column=column, figsize=(12, 6), bins=bins)
    plt.title(title)
    plt.show()

def plot_countplot(df, column, hue=None, title=None, xticks_labels=None):
    """Gera um gráfico de contagem."""
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x=column, hue=hue)
    if title:
        ax.set_title(title)
    # Não vamos mais definir os xticks_labels aqui
    return ax # Retorna o objeto Axes

def plot_correlation_matrix(df_numerics):
    """Gera e exibe a matriz de correlação."""
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_numerics.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Mapa de Correlação entre Variáveis")
    plt.show()

def plot_boxplot(df, x, y, title):
    """Gera um boxplot."""
    sns.boxplot(x=x, y=y, data=df, palette='pastel')
    plt.title(title)
    plt.show()

def plot_boxplot_with_stats(df, x, y, title):
    """Gera um boxplot com linhas de média e mediana."""
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x=x, y=y, data=df, palette='pastel')

    means = df.groupby(x)[y].mean()
    medians = df.groupby(x)[y].median()

    for i in range(len(means)):
        plt.hlines(means.iloc[i], i - 0.4, i + 0.4, colors='red', linestyles='--', label='Média' if i == 0 else "")
        plt.hlines(medians.iloc[i], i - 0.4, i + 0.4, colors='blue', linestyles='-', label='Mediana' if i == 0 else "")

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()

def perform_mannwhitneyu_test(group1, group2):
    """Realiza o teste de Mann-Whitney U e imprime os resultados."""
    statistics, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f'Estatística do teste: {statistics}')
    print(f'Valor-p: {p_value}')

if __name__ == '__main__':
    data_path = 'data/churn.csv'
    df = pd.read_csv(data_path)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    plot_distribution(df, 'tenure', 'Distribuição de Tenure')
    plot_countplot(df, 'Churn', title='Distribuição de Churn (Cancelamento de Serviço)')

    df_numerics = df.select_dtypes(include=['int64', 'float64'])
    plot_correlation_matrix(df_numerics)

    sns.countplot(x="gender", hue="Churn", data=df)
    plt.title('Churn por Gênero')
    plt.xlabel('Gênero')
    plt.ylabel('Churn')
    plt.show()
    
    sns.countplot(x="SeniorCitizen", hue="Churn",data=df)
    plt.title("Churn por Perfil Sênior")
    plt.xticks([0, 1], ['Não Sênior', 'Sênior'])
    plt.xlabel('Cliente Sênior')
    plt.ylabel('Quantidade de Clientes')
    plt.show()

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
    plot_boxplot_with_stats(df, 'Churn', 'QtdServicos', 'Distribuição de Churn por Quantidade de Serviços Contratados')
    plot_countplot(df, 'Contract', hue='Churn', title='Churn por Tipo de Contrato')

    df_clean = df.dropna(subset=['MonthlyCharges', 'TotalCharges']).copy()
    plot_boxplot(df_clean, 'Churn', 'MonthlyCharges', 'Monthly Charges por Churn')
    plot_boxplot(df_clean, 'Churn', 'TotalCharges', 'Total Charges por Churn')

    group_0_qtd_servicos = df[df['Churn'] == 0]['QtdServicos']
    group_1_qtd_servicos = df[df['Churn'] == 1]['QtdServicos']
    perform_mannwhitneyu_test(group_0_qtd_servicos, group_1_qtd_servicos)

    for col in ['MonthlyCharges', 'TotalCharges']:
        group_0 = df_clean.loc[df_clean['Churn'] == 0, col]
        group_1 = df_clean.loc[df_clean['Churn'] == 1, col]
        perform_mannwhitneyu_test(group_0, group_1)