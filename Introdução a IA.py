import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Carregar o dataset Iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Exibir as primeiras linhas do DataFrame
print("Primeiras linhas do dataset: ")
print(iris_df.head())

# Estatísticas Descritivas
print("\nEstatísticas Descritivas: ")
print(iris_df.describe())

# Contagem de cada espécie
print("Contagem de cada espécie: ")
print(iris_df["species"].value_counts())

# Boxplots para cada característica por espécie
plt.figure(figsize=(15, 10))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=iris_df)
plt.suptitle('Boxplots das Características por Espécie')
plt.show()

# Pairplot com mapeamento de cor por espécie
sns.pairplot(iris_df, hue='species')
plt.suptitle('Pairplot das Características com Mapeamento de Cor por Espécie')
plt.show()

# Mapa de calor da correlação entre características
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.drop(columns='species').corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Calor da Correlação entre Características')
plt.show()

# Importação do Dataset Iris
from sklearn.datasets import load_iris
iris = load_iris()

# Pré-processamento: Divisão em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# Exemplo de Regressão Linear
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predições e Visualização
predictions = model.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel("Valores Reais")
plt.ylabel("Predições")
plt.title("Regressão Linear - Predições vs Realidade")
plt.show()

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import numpy as np

# Carregar o dataset Iris
iris = datasets.load_iris()
X = iris.data[:, 0] # Sepal Lenght
Y = iris.data[:, 1] # Sepal Width

# Redimensionar X para 2D (necessário para o scikit-learn)
# X.resshape(-1, 1) transfoprma X de um array unidimensional (por exemplo, [a, b, c] para um array bidimensional com)
# O scikit-learn espera que x(os dados das variaveis independentes) seja sempre um array bidimensional
X = X.reshape(-1, 1)

# Ajustar o modelo de Regressão Linear
model = LinearRegression()
model.fit(X, Y)

# Previsões para a linha de regressão
Y_fit = model.predict(X)

# Plotar os dados e a linha de regressão
plt.scatter(X, Y, color= 'blue' )
plt.plot(X, Y_fit, color= 'red', linewidth=2)
plt.xlabel('Sepal Lenght (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Regressão Linear - Sepal Lenght vs Sepal Width')
plt.show()