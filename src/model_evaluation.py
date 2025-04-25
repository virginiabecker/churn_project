from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo e imprime o relatório de classificação e acurácia."""
    y_pred = model.predict(X_test)
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    return y_pred

def evaluate_model_with_optimized_threshold(model, X_test, y_test):
    """Avalia o modelo usando um threshold otimizado para F1-score."""
    y_probs = model.predict_proba(X_test)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresh[best_idx]
    y_pred_f1 = (y_probs >= best_threshold_f1).astype(int)
    print(f"Threshold ótimo para F1 máximo ({f1_scores[best_idx]:.3f}): {best_threshold_f1:.3f}")
    print("\nRelatório com threshold otimizado para F1:")
    print(classification_report(y_test, y_pred_f1))
    print(f"Acurácia (com threshold otimizado): {accuracy_score(y_test, y_pred_f1):.4f}")
    return y_pred_f1, best_threshold_f1