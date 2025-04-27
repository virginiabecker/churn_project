from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def train_decision_tree(X_train, y_train, random_state=42):
    """Treina um modelo de Árvore de Decisão."""
    tree_model = DecisionTreeClassifier(random_state=random_state)
    tree_model.fit(X_train, y_train)
    return tree_model

def predict_and_evaluate_tree(model, X_test, y_test):
    """Realiza previsões com o modelo de árvore e avalia."""
    y_pred = model.predict(X_test)
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nAcurácia:")
    print(accuracy_score(y_test, y_pred))
    return y_pred

def plot_tree_model(model, feature_names):
    """Plota a árvore de decisão."""
    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=['No Churn', 'Churn'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.show()

def get_feature_importance_tree(model, feature_names):
    """Retorna a importância das features da árvore."""
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=False)
    print("Top 10 variáveis mais importantes:\n", fi.head(10))
    plt.figure(figsize=(8, 6))
    fi.head(10).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Importância das 10 principais features")
    plt.xlabel("Importance")
    plt.show()

def tune_decision_tree(X_train, y_train, param_grid, cv=5, scoring='accuracy'):
    """Ajusta os hiperparâmetros da árvore de decisão."""
    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print("Melhores parâmetros encontrados:", grid.best_params_)
    return grid.best_estimator_

def train_xgboost_model(X_train, y_train, scale_pos_weight=1, random_state=42):
    """Treina o modelo XGBoost."""
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=random_state
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def predict_and_evaluate_xgboost(model, X_test, y_test):
    """Realiza previsões com o modelo XGBoost e avalia."""
    y_pred_xgb = model.predict(X_test)
    print("\nRelatório de Classificação (XGBoost):")
    print(classification_report(y_test, y_pred_xgb))
    print("Acurácia (XGBoost):", accuracy_score(y_test, y_pred_xgb))
    return y_pred_xgb

def tune_xgboost(X_train, y_train, param_grid, scale_pos_weight=1, cv=5, scoring='f1'):
    """Ajusta os hiperparâmetros do XGBoost."""
    xgb = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print("Melhores parâmetros encontrados:", grid.best_params_)
    print("Melhor F1 (CV):", grid.best_score_)
    return grid.best_estimator_

def train_xgboost_with_smote(X_train, y_train, param_grid, cv=5, scoring='f1'):
    """Treina e ajusta o XGBoost com SMOTE."""
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(
            objective='binary:logistic',
            random_state=42
        ))
    ])

    grid_smote = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_smote.fit(X_train, y_train)
    print("Melhores parâmetros (SMOTE):", grid_smote.best_params_)
    print("Melhor F1 (CV):", grid_smote.best_score_)
    return grid_smote.best_estimator_

def evaluate_best_model(model, X_test, y_test):
    """Avalia o melhor modelo e otimiza o threshold para F1."""
    y_probs = model.predict_proba(X_test)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresh[best_idx]
    y_pred_f1 = (y_probs >= best_threshold_f1).astype(int)
    print(f"Threshold ótimo para F1 máximo ({f1_scores[best_idx]:.3f}): {best_threshold_f1:.3f}")
    print("\nRelatório com threshold otimizado para F1:")
    print(classification_report(y_test, y_pred_f1))
    print("Acurácia (Ajustado):", accuracy_score(y_test, y_pred_f1))
    return y_pred_f1

def get_feature_importance_adjusted(model, feature_names):
    """Retorna e plota a importância das features do modelo ajustado."""
    importance = model.feature_importances_
    df_adjusted_importance = pd.DataFrame({'Feature': feature_names, 'Importância': importance})
    df_adjusted_importance = df_adjusted_importance.sort_values(by='Importância', ascending=False).reset_index(drop=True)
    df_adjusted_importance_top10 = df_adjusted_importance.head(10).copy()
    df_adjusted_importance_top10['dummy_hue'] = 'Importance'
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importância', y='Feature', hue='dummy_hue', data=df_adjusted_importance_top10, palette='viridis', legend=False)
    return df_adjusted_importance


