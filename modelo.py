import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pickle

# 1. Criar o dataset sintético
np.random.seed(42)

# Dados fictícios
dados = {
    'idade': np.random.randint(20, 60, size=100),
    'genero': np.random.choice(['masculino', 'feminino', 'outro'], size=100),
    'tempo_trabalho': np.random.randint(1, 30, size=100),
    'setor': np.random.choice(['tecnologia', 'saude', 'educacao'], size=100),
    'salario': np.random.randint(2000, 15000, size=100)  # Salário aleatório entre 2000 e 15000
}

df = pd.DataFrame(dados)

# 2. Preprocessar os dados: converter as variáveis categóricas em números
label_encoder = LabelEncoder()

# Convertendo 'genero' e 'setor' para variáveis numéricas
df['genero'] = label_encoder.fit_transform(df['genero'])
df['setor'] = label_encoder.fit_transform(df['setor'])

# 3. Separando as variáveis independentes (X) e a variável dependente (y)
X = df[['idade', 'genero', 'tempo_trabalho', 'setor']]
y = df['salario']

# 4. Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Criar o modelo de Árvore de Decisão
modelo = DecisionTreeRegressor(random_state=42)

# Treinar o modelo
modelo.fit(X_train, y_train)

# 6. Fazer previsões
y_pred = modelo.predict(X_test)

# 7. Avaliar o modelo: calcular o erro quadrático médio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Erro quadrático médio (MSE): {mse:.2f}")

# Mostrar as primeiras previsões comparadas com os valores reais
comparacao = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
print(comparacao.head())


# Salvando o modelo treinado
with open('modelo_preditivo.pkl', 'wb') as f:
    pickle.dump(modelo, f)