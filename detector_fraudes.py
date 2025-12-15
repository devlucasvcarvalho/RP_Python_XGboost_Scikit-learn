import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import numpy as np

X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.99, 0.01],  # 99% classe 0, 1% classe 1
    random_state=42,
    n_features=20
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=== Desafio 3: Detector de Fraudes ===")
print(f"Distribuição das classes no conjunto completo: {np.bincount(y)}")
print(f"Distribuição no treino: {np.bincount(y_train)}")
print(f"Distribuição no teste: {np.bincount(y_test)}")

# Modelo 1: Sem ajuste para dados desbalanceados
print("\n--- Modelo 1: Sem ajuste ---")
modelo1 = xgb.XGBClassifier(random_state=42)
modelo1.fit(X_train, y_train)
y_pred1 = modelo1.predict(X_test)

print(f"Recall da classe 1 (fraudes): {recall_score(y_test, y_pred1, pos_label=1):.4f}")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred1))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred1))

# Modelo 2: Com scale_pos_weight
print("\n--- Modelo 2: Com scale_pos_weight ---")

negativos = np.sum(y_train == 0)
positivos = np.sum(y_train == 1)
scale_pos_weight = negativos / positivos

print(f"scale_pos_weight calculado: {scale_pos_weight:.2f}")

modelo2 = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
modelo2.fit(X_train, y_train)
y_pred2 = modelo2.predict(X_test)

print(f"Recall da classe 1 (fraudes): {recall_score(y_test, y_pred2, pos_label=1):.4f}")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred2))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred2))