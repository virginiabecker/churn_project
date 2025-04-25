# Previsão de Churn de Clientes

O projeto tem como objetivo, através de uma análise minuciosa, prever a propensão ao cancelamento de serviços (Churn) em uma empresa de telecomunicações. Para isso, foram utilizadas técnicas de análise de dados e machine learning.
O foco principal foi maximizar o recall, buscando reduzir ao máximo os falsos negativos e identificar corretamente o maior número de clientes que realmente cancelariam.


## Objetivo do Projeto

* Analisar o comportamento dos clientes ('Não Churn' e 'Churn') diante das variáveis envolvidas.

* Identificar padrões relacionados ao Churn.

* Construir um modelo preditivo eficiente para retenção de clientes.

* Priorizar recall como métrica principal, visando reduzir perdas.


## Principais Etapas

### 1. Análise Exploratória dos Dados (EDA)

Visualizações, comparações entre grupos e teste estatístico de Mann-Whitney para variáveis numéricas não normais.

### 2. Pré-processamento

* Tratamento de dados faltantes: Substituição dos valores ausentes na coluna TotalCharges pela mediana.

* Criação de novas variáveis: A coluna QtdServicos foi gerada para representar a quantidade de serviços contratados por cada cliente.

* Conversão de variáveis categóricas: As variáveis categóricas foram transformadas em variáveis dummies utilizando pd.get_dummies.

* Remoção de colunas irrelevantes: A coluna customerID foi removida do conjunto de dados.

### 3. Modelagem

* Árvore de Decisão: Utilizou-se o modelo de árvore de decisão (DecisionTreeClassifier) para a classificação de churn.

* Não foi aplicado nenhum ajuste de class_weight, e o modelo foi treinado utilizando os dados processados e balanceados nas divisões de treino e teste.


## Organização do Projeto

churn_project/
├── data/                          # Conjunto de dados
│   └── churn.csv                  # Arquivo de dados
├── src/                           # Scripts do projeto
│   ├── data_loader.py             # Carregamento dos dados
│   ├── preprocessing.py           # Pré-processamento dos dados
│   ├── exploratory_analysis.py    # Análise exploratória
│   ├── model_training.py          # Treinamento de modelos
│   ├── model_evaluation.py        # Avaliação de modelos
│   └── visualization.py           # Visualização dos resultados
├── tests/                         # Testes automatizados
│   ├── test_data_loader.py        # Testes de carregamento dos dados
│   ├── test_preprocessing.py      # Testes de pré-processamento
│   ├── test_model.py              # Testes de treinamento de modelos
├── reports/                       # Relatórios (PDF ou HTML)
├── README.md                      # Este arquivo
└── requirements.txt               # Arquivo de dependências


## Conjunto de Dados

O dataset churn.csv contém registros de clientes, com variáveis demográficas, tipo de serviço contratado e histórico de permanência. A variável alvo (Churn) indica se o cliente cancelou (Yes) ou não (No) o serviço.


## Como Usar

### 1.  **Clonar o repositório:**

[Meu Repositório no GitHub](https://github.com/virginiabecker/churn_project.git)

```bash
git clone https://github.com/virginiabecker/churn_project.git
cd churn_project
```

### 2.  **Criar e ativar o ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Linux/macOS
    venv\Scripts\activate  # No Windows
    ```

### 3.  **Instalar as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Certifique-se de ter gerado o `requirements.txt` corretamente)*

### 4.  **Executar o projeto:**
    ```bash
    python src/main.py
    ```
    *(Se o seu script principal tiver um nome diferente, ajuste o comando)*


## Conjunto de Dados

O dataset utilizado é o `churn.csv`, contendo informações sobre clientes de telecomunicações e se eles cancelaram ou não seus serviços.


## Bibliotecas Utilizadas

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* imbalanced-learn



## Autora

**Virginia Becker**
Economista em transição para a área de dados, apaixonada por resolver problemas com análise, machine learning e boas histórias com dados.

[LinkedIn](https://www.linkedin.com/in/virginiastoquettibecker/) • [Medium](https://medium.com/@virginia.becker)