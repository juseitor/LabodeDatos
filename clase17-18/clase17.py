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

#%%

y_real = []

for i in y:
    y_real.append(modeloknn_alturas.predict([[i]])[0])
    

#%%


mse = mean_squared_error(y,y_real)
print(mse)
