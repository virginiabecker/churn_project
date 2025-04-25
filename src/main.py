from src.data_loader import load_data
from src.preprocessing import preprocess_data, split_data
from src.exploratory_analysis import (
    plot_distribution, plot_countplot, plot_correlation_matrix,
    plot_boxplot, plot_boxplot_with_stats, perform_mannwhitneyu_test
)
from src.model_training import (
    train_decision_tree, tune_decision_tree, train_xgboost_model,
    tune_xgboost, train_xgboost_with_smote, get_feature_importance_adjusted
)
from src.model_evaluation import evaluate_model, evaluate_model_with_optimized_threshold
from src.visualization import plot_tree_model, plot_feature_importance, plot_feature_importance_adjusted
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd  # Importação explícita do pandas

# Caminho para o arquivo de dados
DATA_PATH = 'data/churn.csv'

if __name__ == "__main__":
    # 1. Carregar os dados
    df = load_data(DATA_PATH)
    print("Dados carregados.")

    # 2. Pré-processar os dados
    X, y = preprocess_data(df.copy())
    X_train, X_test, y_train, y_test = split_data(X, y, stratify=y)
    print("Dados pré-processados e divididos.")

    # 3. Análise exploratória (você pode comentar/descomentar o que quiser executar)
    print("\n--- Análise Exploratória ---")
    plot_distribution(df, 'tenure', 'Distribuição de Tenure')
    plot_countplot(df, 'Churn', title='Distribuição de Churn')
    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    plot_correlation_matrix(df_numeric)
    plot_countplot(df, 'gender', hue='Churn', title='Churn por Gênero')
    plot_countplot(df, 'SeniorCitizen', hue='Churn', title="Churn por Perfil Sênior", xticks_labels=['Não Sênior', 'Sênior'])
    plot_boxplot_with_stats(df, 'Churn', 'QtdServicos', 'Churn por Qtd de Serviços')
    plot_countplot(df, 'Contract', hue='Churn', title='Churn por Tipo de Contrato')
    df_clean = df.dropna(subset=['MonthlyCharges', 'TotalCharges']).copy()
    plot_boxplot(df_clean, 'Churn', 'MonthlyCharges', 'Monthly Charges por Churn')
    plot_boxplot(df_clean, 'Churn', 'TotalCharges', 'Total Charges por Churn')
    group_0_qtd = df[df['Churn'] == 0]['QtdServicos']
    group_1_qtd = df[df['Churn'] == 1]['QtdServicos']
    perform_mannwhitneyu_test(group_0_qtd, group_1_qtd, 'QtdServicos')
    for col in ['MonthlyCharges', 'TotalCharges']:
        group_0 = df_clean.loc[df_clean['Churn'] == 0, col]
        group_1 = df_clean.loc[df_clean['Churn'] == 1, col]
        perform_mannwhitneyu_test(group_0, group_1, col)

    # 4. Treinar e avaliar modelos
    print("\n--- Treinamento e Avaliação de Modelos ---")

    # 4.1. Árvore de Decisão
    print("\n--- Árvore de Decisão ---")
    tree_model = train_decision_tree(X_train, y_train)
    evaluate_model(tree_model, X_test, y_test)
    plot_tree_model(tree_model, X_train.columns, ['No Churn', 'Churn'], title='Árvore de Decisão Inicial')
    feature_importance_tree = pd.Series(tree_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plot_feature_importance(feature_importance_tree, title='Importância das Features (Árvore de Decisão)')

    param_grid_tree = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_tree_model = tune_decision_tree(X_train, y_train, param_grid_tree, scoring='f1')
    print("\n--- Melhor Árvore de Decisão (Ajustada) ---")
    evaluate_model(best_tree_model, X_test, y_test)
    plot_tree_model(best_tree_model, X_train.columns, ['No Churn', 'Churn'], title='Melhor Árvore de Decisão (Ajustada)')
    feature_importance_best_tree = pd.Series(best_tree_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plot_feature_importance(feature_importance_best_tree, title='Importância das Features (Melhor Árvore de Decisão)')

    # 4.2. XGBoost
    print("\n--- XGBoost ---")
    scale_pos_weight_xgb = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_model = train_xgboost_model(X_train, y_train, scale_pos_weight=scale_pos_weight_xgb)
    evaluate_model(xgb_model, X_test, y_test)
    feature_importance_xgb = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plot_feature_importance(feature_importance_xgb, title='Importância das Features (XGBoost)')

    param_grid_xgb = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    best_xgb_model = tune_xgboost(X_train, y_train, param_grid_xgb, scale_pos_weight=scale_pos_weight_xgb, scoring='f1')
    print("\n--- Melhor XGBoost (Ajustado) ---")
    evaluate_model(best_xgb_model, X_test, y_test)
    adjusted_importance_xgb = get_feature_importance_adjusted(best_xgb_model, X_train.columns)
    plot_feature_importance_adjusted(adjusted_importance_xgb, title='Importância das 10 Principais Features (Melhor XGBoost)')

    # 4.3. XGBoost com SMOTE
    print("\n--- Melhor XGBoost com SMOTE ---")
    param_grid_xgb_smote = {
        'xgb__max_depth': [5, 7],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__n_estimators': [100, 200],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }
    best_xgb_smote_model = train_xgboost_with_smote(X_train, y_train, param_grid_xgb_smote, scoring='f1')
    evaluate_model_with_optimized_threshold(best_xgb_smote_model, X_test, y_test)
    adjusted_importance_xgb_smote = get_feature_importance_adjusted(best_xgb_smote_model.named_steps['xgb'], X_train.columns)
    plot_feature_importance_adjusted(adjusted_importance_xgb_smote, title='Importância das 10 Principais Features (Melhor XGBoost com SMOTE)')

    # 5. Árvore Didática
    print("\n--- Árvore Didática ---")
    top_n_features = adjusted_importance_xgb_smote['Feature'].head(3).tolist()
    X_train_small = X_train[top_n_features]
    didactic_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    didactic_tree.fit(X_train_small, y_train)
    plot_tree_model(didactic_tree, top_n_features, ['No Churn', 'Churn'], title='Árvore Didática (Top 3 Features)')
    print("\n--- Árvore Didática (Textual) ---")
    print(export_text(didactic_tree, feature_names=top_n_features))

    print("\nFluxo principal do projeto concluído!")