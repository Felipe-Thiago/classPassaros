"""
    This dataset is provided by Dr. D. Liu of Beijing Museum of Natural History.

    Fatec Votorantim
    Alunos: Felipe Thiago da Silva e Yara Bona de Paes
    Classificação de pássaros por formação dos ossos e hábitos de vida
"""
#%% Importar Bibliotecas
# python -m pip install scikit-learn
# python -m pip install keras
# python -m pip install tensorflow
# python -m pip install matplotlib
# python -m pip install pandas

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

'''
Variáveis de entrada (features):
    - 1/2: Length and Diameter of Humerus (Comprimento e diâmetro do úmero)
    - 3/4: Length and Diameter of Ulna (Comprimento e diâmetro do rádio)
    - 5/6: Length and Diameter of Femur (Comprimento e diâmetro do fêmur)
    - 7/8: Lenght and Diameter of Tibiotarsus (Comprimento e diâmetro do tíbio)
    - 9/10: Length and Diameter of Tarsometatarsus (Comprimento e diâmetro do tarso-metatarso)
Todas as variáveis de entrada são numéricas e contínuas, representando o comprimento e o diâmetro dos ossos dos pássaros em milímetros (mm).

Variáveis de saída (classes):
    - 1: Swimming Birds (Pássaros nadadores) -      SW
    - 2: Wading Birds (Pássaros de água) -          W
    - 3: Terrestrial Birds (Pássaros terrestres) -  T
    - 4: Raptor Birds (Pássaros de rapina) -        R
    - 5: Scansorial Birds (Pássaros escansoriais) - P
    - 6: Singing Birds (Pássaros cantores) -        SO

    https://www.kaggle.com/datasets/zhangjuefei/birds-bones-and-living-habits/data
'''

#%% Carregar base de dados
df = pd.read_csv("bird.csv")

# Definir variável de entrada
X = df.loc[:, 'huml':'tarw']  # de altura do úmero até comprimento do tarso-metatarso

# Definir variável de saída
y1 = df['type'].values.reshape(-1, 1)  # vetor de classes 

ohe = OneHotEncoder(sparse_output=False)  # Inicializar o codificador one-hot
y = ohe.fit_transform(y1)


#%% Separar em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
escala = StandardScaler()
X_train = escala.fit_transform(X_train)
X_test = escala.transform(X_test)



#%% Construir o modelo
mlp = Sequential()
mlp.add(Input(shape=(10,))) # 10 features de entrada (huml até tarw)
mlp.add(Dense(64, activation='sigmoid')) # Camada oculta 64 neurons 
#mlp.add(Dense(10, activation='sigmoid')) # Segunda camada oculta
# --------------------------------------------------------------------------- 64 neurons activation='relu'
mlp.add(Dense(6, activation='softmax')) # Camada de saída com 6 classes (SW, W, T, R, P, SO)

# Compilar o modelo
mlp.compile(
    loss='categorical_crossentropy', 
    optimizer=Adam(learning_rate=0.001),  # ---------------------------------------------------- learning_rate=0.001
    metrics=['accuracy'])

callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.005,
    verbose=1,
    mode='min',
)

# Treinar o modelo
mlp.fit(X_train, y_train, epochs=200,  # ------------------------------------------- epochs=200
        batch_size=32, validation_data=(X_test, y_test), 
        #callbacks=[callback],
        verbose=1)



#%% Avaliar o modelo
loss, accuracy = mlp.evaluate(X_test, y_test)
print(f"\nAcurácia: {accuracy:.4f}")



#%% Fazer previsões para um pássaro desconhecido
#X_conh = np.array([[77.65, 5.7, 65.76, 4.77, 40.04, 3.52, 69.17, 3.4, 35.78, 3.41]]) # dados reais para um SW
X_desc = np.array([[ 88.0, 6.5, 70.2, 5.5, 48.07, 4.0, 66.0, 5.99, 32.3, 3.55]],) # dados fictícios
y_desc = mlp.predict(X_desc)

print('\nPássaro previsto: ', y_desc)
classe_idx = np.argmax(y_desc)  # Obter o índice da classe prevista
classe_desc = ohe.categories_[0][classe_idx]  # Obter a classe prevista
prob = y_desc[0, classe_idx]  # Obter a probabilidade da classe prevista
print(f"Pássaro desconhecido previsto: {classe_desc} \n Probabilidade: {prob:.4f}")
input('Aperte uma tecla para continuar:')

# Prever as classes para o conjunto de teste
y_prev = mlp.predict(X_test)
y_prev_class = np.argmax(y_prev, axis=1)  # Obter a classe prevista para cada entrada
y_test_class = np.argmax(y_test, axis=1)  # Obter a classe real para cada entrada

# Calcular a matriz de confusão
'''
conf_mat = confusion_matrix(y_test_class, y_prev_class)
print("Matriz de confusão:\n", conf_mat)
'''

class_names = ohe.categories_[0]  # Obter os nomes das classes
print("\nClasses:", class_names)
input('Aperte uma tecla para continuar:')
cm = confusion_matrix(y_test_class, y_prev_class, labels=range(len(class_names)))

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
display.plot()
plt.show()