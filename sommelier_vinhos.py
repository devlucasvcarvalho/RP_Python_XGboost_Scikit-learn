import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=100,
    random_state=42
)

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print("=== Desafio 2: Sommelier de Vinhos ===")
print(f"Acurácia: {acuracia:.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# matriz de confusão
matriz_confusao = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(matriz_confusao)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.title('Matriz de Confusão - Classificação de Vinhos')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.tight_layout()
plt.show()