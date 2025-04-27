import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import pandas as pd

def plot_tree_model(model, feature_names, class_names, title='Decision Tree'):
    """Plota a árvore de decisão."""
    plt.figure(figsize=(20, 10))
    plot_tree(model,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title(title)
    plt.show()

def plot_feature_importance(feature_importance, top_n=10, title='Feature Importance'):
    """Plota as top N features mais importantes."""
    plt.figure(figsize=(8, 6))
    feature_importance.head(top_n).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()



def plot_feature_importance_adjusted(importance_df, title='Importância das 10 Principais Features (Modelo Ajustado)'):
    """Plota a importância das features ajustada."""
    if importance_df is not None and not importance_df.empty:
        df_adjusted_importance_top10 = importance_df.head(10).copy()
        df_adjusted_importance_top10['dummy_hue'] = 'Importance'

    else:
        print("Erro: O DataFrame de importância está vazio ou é None.")
    df_adjusted_importance_top10 = importance_df.head(10).copy()
    df_adjusted_importance_top10['dummy_hue'] = 'Importance'
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importância', y='Feature', hue='dummy_hue', data=df_adjusted_importance_top10, palette='viridis', legend=False)
    plt.title(title)
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()