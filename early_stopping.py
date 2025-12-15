import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=== Desafio 5: Early Stopping e Salvamento ===")

modelo = xgb.XGBClassifier(
    n_estimators=1000,  # Número máximo de árvores
    learning_rate=0.1,
    random_state=42
)

print("Treinando modelo com early stopping...")
modelo.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # Conjunto de validação
    eval_metric="logloss",  # Métrica para monitorar
    early_stopping_rounds=10,  # Parar se não melhorar em 10 rodadas
    verbose=False  # Não mostrar mensagens durante treino
)

print(f"Treinamento interrompido na iteração: {modelo.best_iteration}")
print(f"Melhor score: {modelo.best_score:.4f}")

y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {acuracia:.4f}")

nome_arquivo = "meu_modelo.json"
modelo.save_model(nome_arquivo)
print(f"\nModelo salvo em: {nome_arquivo}")
print(f"Tamanho do arquivo: {os.path.getsize(nome_arquivo)} bytes")

print("\nCarregando modelo salvo...")
modelo_carregado = xgb.XGBClassifier()
modelo_carregado.load_model(nome_arquivo)

y_pred_carregado = modelo_carregado.predict(X_test)
acuracia_carregado = accuracy_score(y_test, y_pred_carregado)

print(f"Acurácia com modelo carregado: {acuracia_carregado:.4f}")

print(f"As previsões são iguais? {(y_pred == y_pred_carregado).all()}")

if os.path.exists(nome_arquivo):
    os.remove(nome_arquivo)
    print(f"\nArquivo {nome_arquivo} removido.")