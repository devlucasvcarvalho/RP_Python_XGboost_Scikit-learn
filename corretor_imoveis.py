import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo XGBRegressor
modelo = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    random_state=42
)

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

rmse_dolares = rmse * 100000
mae_dolares = mae * 100000

print("=== Desafio 1: Corretor de Imóveis ===")
print(f"RMSE: {rmse:.4f} (${rmse_dolares:.2f})")
print(f"MAE: {mae:.4f} (${mae_dolares:.2f})")
print(f"Erro médio absoluto em dólares: ${mae_dolares:.2f}")