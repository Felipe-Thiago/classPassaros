"""
    This dataset is provided by Dr. D. Liu of Beijing Museum of Natural History.

    Fatec Votorantim
    Alunos: Felipe Thiago da Silva e Yara Bona de Paes
    Classificação de pássaros por formação dos ossos e hábitos de vida
"""

#%% Bibliotecas

# python -m pip install scikit-learn
# python -m pip install kagglehub[pandas-datasets]
# python -m pip install matplotlib
# python -m pip install pandas 

import kagglehub as khub # dataset utilizado
import pandas as pd
from sklearn.neural_network import MLPClassifier # classificador de rede neural

from sklearn.model_selection import train_test_split   # separa treinamento e teste
from sklearn.metrics import accuracy_score

import os # para manipulação de arquivos e diretórios

#%% Carga dos dados
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
# Download do dataset na nuvem Kaggle (OBS.: possui dados nulos, ver tabela no link acima)
# path = khub.dataset_download("zhangjuefei/birds-bones-and-living-habits")
# print("Path to dataset files:", path)

script_dir = os.path.dirname(os.path.abspath(__file__))  # diretório do script
csv_path = os.path.join(script_dir, "bird.csv")  # caminho do arquivo CSV
df = pd.read_csv(csv_path) # baixado, completo e salvo na pasta ClassPassaros
print("Primeiros 5 registros\n", df.head())
input("Aperte uma tecla para continuar: \n")




#%% Seleção dos dados
X = df.loc[:, 'huml':'tarw']   # de altura do úmero até comprimento do tarso-metatarso
y = df.loc[:, 'type'] # vetor de classes

print("Matriz de entradas (treinamento):\n", X) 
input("Aperte uma tecla para continuar: \n")


print("Vetor de classes (treinamento):\n", y)
input("Aperte uma tecla para continuar: \n")


#%% Separação dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
print("\nFormato dados de treinamento", X_train.shape, y_train.shape)
print("Formato dados de teste", X_test.shape, y_test.shape)
input('Aperte uma tecla para continuar:')

#%% Configuração da rede neural
mlp = MLPClassifier(verbose=True,    # default=False imprime mensagens de progresso
                    hidden_layer_sizes=(100,),  # default (100,)
                    max_iter=400,   # default=200
                    tol=1e-6,       # default=1e-4
                    activation='relu',   # default 'relu'
                    learning_rate='constant',) # default 'constant'

                    # n_iter_no_changeint, default=10
                    # solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
                    # batch_size=<N_int>, default=’auto’
                    # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
                    # early_stopping=<bool>, default=False

#%% Treinamento da rede
#mlp.fit(X,y)
mlp.fit(X_train, y_train)

#%% Teste
#entrada = pd.DataFrame([[ 88.0, 6.5, 70.2, 5.5, 48.07, 4.0, 66.0, 5.99, 32.3, 3.55]],) # dados ficticios
entrada = pd.DataFrame([[ 77.65, 5.7, 65.76, 4.77, 40.04, 3.52, 69.17, 3.4, 35.78, 3.41]], columns=X.columns) # dados reais para um SW
print('\nPássaro previsto: ', mlp.predict(entrada)) # dados reais
input('Aperte uma tecla para continuar:')

#%% DESEMPENHO SOBRE O CONJUNTO DE TESTE
# previsões
y_pred = mlp.predict(X_test)
print(y_pred.dtype, ' vetor de previsoes = ', y_pred)
input('Aperte uma tecla para continuar:')

# desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print("\nAcurácia:", accuracy)   # soma dos acertos / nº dados de teste
print("Erro = ", mlp.loss_)    # fator de perda (erro)
input('Aperte uma tecla para continuar:')


# -----------------------------------------------------------------------------------------------------------------


#%% MATRIZ DE CONFUSÃO
from sklearn.metrics import confusion_matrix

# Calcular a matriz de confusão
matriz_confusao = confusion_matrix(y_test, y_pred)

# Imprimir a matriz de confusão
print("Matriz de Confusão:")
print(matriz_confusao)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=mlp.classes_)

display = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=mlp.classes_)
display.plot()
plt.show()

# -----------------------------------------------------------------------------------------------------------------

for tam_camada_oculta in [(20,), (50,), (100,), (200,), (10, 10), (20, 20), (50, 50)]:
    mlp = MLPClassifier(hidden_layer_sizes = tam_camada_oculta, 
                    max_iter=2000,   # default=200
                    tol=1e-6,       # default=1e-4
                    learning_rate='adaptive', # default 'constant'
                    activation='relu')   # default 'relu' estava 'logistic'
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAcurácia com ", mlp.hidden_layer_sizes, " neurons:", accuracy)

# -----------------------------------------------------------------------------------------------------------------
    #%% Parâmetros da rede
print("\nClasses = ", mlp.classes_)     # lista de classes
print("Erro = ", mlp.loss_)    # fator de perda (erro)
print("Amostras visitadas = ", mlp.t_)     # número de amostras de treinamento visitadas 
print("Atributos de entrada = ", mlp.n_features_in_)   # número de atributos de entrada (campos de X)
print("N ciclos = ", mlp.n_iter_)      # númerode iterações no treinamento
print("N de camadas = ", mlp.n_layers_)    # número de camadas da rede
print("Tamanhos das camadas ocultas: ", mlp.hidden_layer_sizes)
print("N de neurons saida = ", mlp.n_outputs_)   # número de neurons de saida
print("F de ativação = ", mlp.out_activation_)  # função de ativação utilizada





