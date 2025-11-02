import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#%% Abrimos y lipiamos df_alturas

df_alturas = pd.read_csv('/home/jusa/LabodeDatos/clase17-18/archivosclase17-18/2025C2 - Alturas - Hoja 1.csv')

df_alturas.drop(columns=['¿Sabes quien mas mide 156?', 'Mi mamiiii', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], inplace = True)
df_alturas.dropna(subset = ['LU', 'Altura (cm)', 'Sexo', 'Mami'], how = 'any' ,inplace = True)

#%% Hago grafico para visualizar los datos

fig, ax = plt.subplots()

sns.scatterplot(
    data = df_alturas,
    x = 'Mami',
    y = 'Altura (cm)',
    hue = 'Sexo'
    )

ax.set_xlabel("Altura de la Madre (cm)")

#%%

y = df_alturas['Altura (cm)'].values
x = df_alturas[['Mami']].values

#%%

modeloknn_alturas = KNeighborsRegressor(
    n_neighbors=5, # '#' de vecinos que se consideran para predecir
    weights = 'distance', #Cómo se pesan los vecinos para hacer la predicción (ó 'uniform')
    algorithm = 'auto' #Algoritmo utilizado para encontrar los vecinos:
    )
modeloknn_alturas.fit(x,y)

#%%

y_pred = modeloknn_alturas.predict([[160]])
print("La altura predicha es de " + str(y_pred[0]))

#%%Este ciclo es para probar cuanto mejora el mse ir sumando
#cantidad de Neighbors

mses = []
n = 1


while n <= 20:
    modeloknn_loop = KNeighborsRegressor(
        n_neighbors=n, 
        weights = 'distance', 
        algorithm = 'auto' 
        )
    modeloknn_loop.fit(x,y)
    y_pred = []
    for i in y:
        y_pred.append(modeloknn_loop.predict([[i]])[0])
    mse = mean_squared_error(y,y_pred)
    mses.append([n,mse])
    n = n + 1
df_mses = pd.DataFrame(mses,columns=["K", "MSE"])

#%% Se lo nota malisimo jej pero tambien son malos los datos dados

fig, ax = plt.subplots()

sns.lineplot(
    data = df_mses,
    x = 'K',
    y = 'MSE'
    )

ax.grid(True)
ax.set_xlabel("Numero de vecinos (k)")
ax.set_ylabel("MSE")
ax.set_title("MSE vs Número de Vecinos (k)")

#%%










#%% Datos de auus MPG (diapo 21/27)

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

df_auto = pd.read_csv('~/LabodeDatos/clase17-18/archivosclase17-18/auto-mpg.xls')

#%%

x = df_auto[['acceleration']]
y = df_auto['mpg']

# división en train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)

#%%

mses = []
n = 1


while n <= 20:
    regresionknn_mpg = KNeighborsRegressor(
        n_neighbors=n, 
        weights = 'distance', 
        algorithm = 'auto' 
        )
    regresionknn_mpg.fit(X_train,y_train)
    y_pred = []
    for i in y:
        y_pred.append(regresionknn_mpg.predict([[i]])[0])
    mse = mean_squared_error(y,y_pred)
    mses.append([n,mse])
    n = n + 1
df_mpg_mses = pd.DataFrame(mses,columns=["K", "MSE"])

#%%
#Aca vemos que el numero de vecinos que hace mas chico el mse es 18
fig, ax = plt.subplots()

sns.lineplot(
    data = df_mpg_mses,
    x = 'K',
    y = 'MSE'
    )

ax.grid(True)
ax.set_xlabel("Numero de vecinos (k)")
ax.set_ylabel("MSE")
ax.set_title("MSE vs Número de Vecinos (k)")
ax.set_xticks(range(0,21,1))








#%% Ahora intentamos hacerlo pero con mas variables

df_mpg = pd.read_csv("~/LabodeDatos/clase17-18/archivosclase17-18/auto-mpg.xls")

X = df_mpg[['weight', 'displacement', 'horsepower', 'acceleration']]
y = df_mpg['mpg']

# división en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

#%%

#NO HACIA FALTA HACER EL LOOP DENTRO. Con hacer
#modeloknn.predict(X_train) aunque X_train sea un Dataframe
#de 4 columnas y 294 filas. Yo pensaba que se debía hacer uno a uno.
valores_k = [1, 3, 5, 10, 20, 30]

resultados = []

for k in valores_k:
    modelo = KNeighborsRegressor(n_neighbors=k)
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    resultados.append({
        'k': k,
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Train_MAE': mean_absolute_error(y_train, y_pred_train),
        'Test_MAE': mean_absolute_error(y_test, y_pred_test),
    })

# Mostrar resultados
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

#%%

plt.figure(figsize=(8, 5))
plt.plot(valores_k, df_resultados['Train_RMSE'], marker='o', label='Train')
plt.plot(valores_k, df_resultados['Test_RMSE'], marker='o', label='Test')

plt.xlabel('Número de vecinos (k)')
plt.ylabel('RMSE')
plt.title('Error cuadrático medio (RMSE) según k en KNN')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
