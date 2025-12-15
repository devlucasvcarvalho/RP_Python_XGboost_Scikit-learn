import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=== Desafio 4: Duelo de Modelos ===")

print("\n--- Árvore de Decisão (padrão) ---")
arvore_padrao = DecisionTreeClassifier(random_state=42)
arvore_padrao.fit(X_train, y_train)
y_pred_arvore_padrao = arvore_padrao.predict(X_test)
acuracia_arvore_padrao = accuracy_score(y_test, y_pred_arvore_padrao)
print(f"Acurácia: {acuracia_arvore_padrao:.4f}")

print("\n--- XGBoost (padrão) ---")
xgb_padrao = xgb.XGBClassifier(random_state=42)
xgb_padrao.fit(X_train, y_train)
y_pred_xgb_padrao = xgb_padrao.predict(X_test)
acuracia_xgb_padrao = accuracy_score(y_test, y_pred_xgb_padrao)
print(f"Acurácia: {acuracia_xgb_padrao:.4f}")

print("\n--- Árvore de Decisão (max_depth=3) ---")
arvore_limitada = DecisionTreeClassifier(max_depth=3, random_state=42)
arvore_limitada.fit(X_train, y_train)
y_pred_arvore_limitada = arvore_limitada.predict(X_test)
acuracia_arvore_limitada = accuracy_score(y_test, y_pred_arvore_limitada)
print(f"Acurácia: {acuracia_arvore_limitada:.4f}")

print("\n--- XGBoost (max_depth=3) ---")
xgb_limitado = xgb.XGBClassifier(max_depth=3, random_state=42)
xgb_limitado.fit(X_train, y_train)
y_pred_xgb_limitado = xgb_limitado.predict(X_test)
acuracia_xgb_limitado = accuracy_score(y_test, y_pred_xgb_limitado)
print(f"Acurácia: {acuracia_xgb_limitado:.4f}")

print("\n=== RESUMO COMPARATIVO ===")
print(f"Árvore de Decisão (padrão): {acuracia_arvore_padrao:.4f}")
print(f"XGBoost (padrão):           {acuracia_xgb_padrao:.4f}")
print(f"Diferença: {abs(acuracia_arvore_padrao - acuracia_xgb_padrao):.4f}")
print(f"\nCom max_depth=3:")
print(f"Árvore de Decisão: {acuracia_arvore_limitada:.4f}")
print(f"XGBoost:           {acuracia_xgb_limitado:.4f}")
print(f"Diferença: {abs(acuracia_arvore_limitada - acuracia_xgb_limitado):.4f}")