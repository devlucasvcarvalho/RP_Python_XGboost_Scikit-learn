# Reconhecimento de Padrões - Atividade Final

Esta atividade prática da disciplina de **Reconhecimento de Padrões** demonstra a aplicação de algoritmos de aprendizado de máquina usando XGBoost em diferentes cenários reais.

## Objetivo
Implementar e comparar diferentes técnicas de aprendizado de máquina utilizando o framework XGBoost para resolver problemas de classificação, regressão e detecção de anomalias.


### **Desafio 1: O Corretor de Imóveis**
**Tipo:** Regressão  
**Problema:** Prever preços de imóveis na Califórnia  
**Modelo:** XGBRegressor  
**Métrica:** RMSE (Root Mean Squared Error)  
**Dataset:** `fetch_california_housing`

### **Desafio 2: O Sommelier de Vinhos**
**Tipo:** Classificação Multiclasse  
**Problema:** Classificar vinhos em 3 categorias  
**Modelo:** XGBClassifier  
**Métrica:** Matriz de Confusão  
**Dataset:** `load_wine`

### **Desafio 3: O Detector de Fraudes**
**Tipo:** Dados Desbalanceados  
**Problema:** Detectar transações fraudulentas  
**Modelo:** XGBClassifier com `scale_pos_weight`  
**Métrica:** Recall da classe minoritária  
**Dataset:** Dados sintéticos desbalanceados

### **Desafio 4: Duelo de Modelos**
**Tipo:** Comparação de Algoritmos  
**Problema:** Comparar XGBoost vs Árvore de Decisão  
**Modelos:** XGBClassifier vs DecisionTreeClassifier  
**Dataset:** `load_breast_cancer`

### **Desafio 5: Early Stopping e Salvamento**
**Tipo:** Engenharia de ML  
**Técnicas:** Early stopping e serialização de modelos  
**Funcionalidade:** Otimização e persistência de modelos treinados

### Instalação
```bash
pip install xgboost scikit-learn matplotlib seaborn numpy