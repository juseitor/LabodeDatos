import numpy as np
import pandas as pd
import duckdb as db
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

#%%

df = pd.read_csv('~/LabodeDatos/clase17-18/archivosclase17-18/datos_libreta_156.txt', sep = " ")

x = np.linspace(min(df['RU']), max(df['RU']))

#%%

fig, ax = plt.subplots()

sns.scatterplot(
    data = df,
    x = 'RU',
    y = 'ID',
    )

ax.set_title("RU vs ID en prueba")







#%% ASI ESTA HECHO POR MI EN CLASE

CONSULTA_PROMEDIO = """
    SELECT AVG(RU) AS "Promedio RU", AVG(ID) AS "Promedio ID"
    FROM df

"""
consulta_promedio = db.query(CONSULTA_PROMEDIO).df()

ru_promedio = consulta_promedio.iloc[0,0]

id_promedio = consulta_promedio.iloc[0,1]

b_cociente = 0

for i in range(len(df)):
    ru_actual = df.iloc[i,0]
    id_actual = df.iloc[i,1]
    b_cociente = b_cociente + (ru_actual - ru_promedio) * (id_actual - id_promedio)
    
b_denominador = 0

for i in range(len(df)):
    ru_actual = df.iloc[i,0]
    id_actual = df.iloc[i,1]
    b_denominador = b_denominador + (ru_actual - ru_promedio) * (ru_actual - ru_promedio)

b1 = b_cociente / b_denominador

b0 = id_promedio - b1 * ru_promedio

#%% Y ASI HECHO POR LA PROFE

#Esta funcion crea un array de cantidad equidistante de valores
#entre los valores pasados como argumento (del min al max)
X = np.linspace(min(df['RU']), max(df['RU']))
a = 100 ### probar con otros valores
b = 0.06 ### probar con otros valores
Y = a + b*X

plt.plot(X, Y,  'r')
plt.show()

xbar = np.mean(X) #mean calcula el promedio de los valores de un array
ybar = np.mean(Y)
b1 = sum((X-xbar)*(Y-ybar))/sum((X-xbar)*(X-xbar))
b0 = ybar - b1*xbar

plt.scatter(df['RU'], df['ID'])
X = np.linspace(min(df['RU']), max(df['RU']))
Y = b0 + b1*X

plt.plot(X, Y,  'r')
plt.show()












#%% coeficientes utilizando sklearn

modelo_lineal = LinearRegression()
# Los argumentos pasados son las columnas del df para entrenar 
#el modelo
modelo_lineal.fit(df[['RU']],df['ID']) 
# Los siguientes son los parametros de la recta que mejor se 
#ajustan a nuestros datos
print(modelo_lineal.intercept_)
print(modelo_lineal.coef_)
## El siguiente es el r2
modelo_lineal.score(df[['RU']],df['ID']) 

#%%

#La recta de regresión es la siguiente

Y = modelo_lineal.intercept_ + modelo_lineal.coef_*x

#Notese que x es un array 

plt.scatter(df['RU'], df['ID'])
plt.plot(X, Y, 'black')

plt.show()

#%%
# Y_pred son los valores que toma la recta sobre los valores RU de la muestra
Y_pred =modelo_lineal.intercept_ + modelo_lineal.coef_*df['RU']

r2 = r2_score(df['ID'], Y_pred)
print("R²: " + str(r2))














#%% Anascombe

df = sns.load_dataset("anscombe")

sns.lmplot(
    data=df, 
    x="x", 
    y="y", 
    col="dataset", 
    hue="dataset",
    col_wrap=2, #En vez de una fila de 4, hace dos filas de graficos de dos
    palette="muted", 
    ci=None, #quita el sombreado
    height=4, 
    scatter_kws={"s": 50, "alpha": 1}
)

#%% primer dataset de anscombe

df1 = df[df['dataset'] == "I"]

#%%

mod_lineal_df1 = LinearRegression()
# De alguna manera el parametro de entrada de fit hay que
#largarlo con doble corchete para elegir la columna. Lo mismo
#con modelo_lineal.score()
mod_lineal_df1.fit(df1[['x']], df1[['y']])

#%%

R12 = mod_lineal_df1.score(df1[['x']], df1[['y']])

#%%

X1 = np.linspace(min(df1['x']), max(df1['x']))

#%%

#b1 = mod_lineal_df1.coef_
#b0 = mod_lineal_df1.intercept_

# Insertando en .coef_[0], nos garantizamos que el Array 
#resultante de Y1 sea de [50] en vez de  [1,50] 
Y1 = mod_lineal_df1.intercept_ + mod_lineal_df1.coef_[0]*X1

# Esta es la recta estimada con los valores de la muestra

#%%

# Graficamos
plt.scatter(df1['x'], df1['y'])
plt.plot(X1, Y1, 'k')
plt.show()

#%%

# Y_pred son los valores que toma la recta sobre los valores 'x' de la muestra
Y_pred =mod_lineal_df1.intercept_ + mod_lineal_df1.coef_[0]*df1['x']

plt.scatter(df1['x'], Y_pred)
plt.plot(df1['x'], Y_pred)

#%%

r2 = r2_score(df1['y'], Y_pred) #Es de sklearn.metrics
print("R²: " + str(r2))

#%%

#Diferencias entre usar r2_score y .score()

 # modelo_lineal.score()
# Usa el modelo ya entrenado.
# Usa los mismos valores que se entrenaron para medir el R².
# Es Rápido,  se usa cuando ya tenés el modelo entrenado 
#y querés el R².
# Se usan los mismos parametros de .fit()

 # r2_score(), Métrica directa
# Comparás valores reales (df1['y']) con tus predicciones (Y_pred).
# No depende del modelo.
# Es más flexible.



#%% segundo dataset de anscombe

df2 = df[df['dataset'] == 'II']

#%%

mod_lineal_df2 = LinearRegression()
mod_lineal_df2.fit(df2[['x']], df2[['y']])

#%%

R22 = mod_lineal_df2.score(df2[['x']], df2[['y']])

#%%

Y1 = mod_lineal_df2.intercept_ + mod_lineal_df2.coef_[0]*df2['x']

plt.scatter(x=df2['x'], y=df2['y'])
plt.plot(df2['x'], Y1)

#%% 3er dataset anscombe

df3 = df[df['dataset'] == 'III']

#%%

mod_lineal_df3 = LinearRegression()
mod_lineal_df3.fit(df3[['x']],df3[['y']])
R32 = mod_lineal_df3.score(df3[['x']],df3[['y']])

#%%

# Recta ajustada de Regresión
Y3_pred = mod_lineal_df3.intercept_ + mod_lineal_df3.coef_[0]*df3['x']

plt.scatter(df3['x'],df3['y'])
plt.plot(df3['x'], Y3_pred)

#%% cuarto dataset anscombe

df4 = df[df['dataset'] == 'IV']

#%%

mod_lineal_df4 = LinearRegression()
mod_lineal_df4.fit(df4[['x']],df4[['y']])
R42 = mod_lineal_df4.score(df4[['x']], df4[['y']])

#%%

Y4_pred = mod_lineal_df4.intercept_ + mod_lineal_df4.coef_[0]*df4['x']

plt.scatter(df4['x'], df4['y'])
plt.plot(df4['x'], Y4_pred)
plt.xticks(range(0,20,1))

#%%









#%% Autos

import numpy as np
import pandas as pd
import duckdb as db
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

df_mpg = pd.read_csv("~/LabodeDatos/clase17-18/archivosclase17-18/auto-mpg.xls")

#%%

mod_lineal_weight = LinearRegression()
mod_lineal_weight.fit(df_mpg[['weight']], df_mpg[['mpg']])
R2_w = mod_lineal_weight.score(df_mpg[['weight']], df_mpg[['mpg']])

#%%

Y_pred = mod_lineal_weight.intercept_ + mod_lineal_weight.coef_[0]*df_mpg['weight']

plt.scatter(df_mpg['weight'], df_mpg['mpg'])
plt.plot(df_mpg['weight'], Y_pred, 'k')

#%%
# Cambio de consumo estimado si aumenta el peso en 100u
print(str(mod_lineal_weight.coef_[0]*100))
# DEF PENDIENTE: Por cada unidad adicional de weight, se
#observa un decrecimiento de (valor pendiente) unidades. 
#A eso lo multiplicamos por 100

#%%











#%% modelo lineal con dos variables

modelo_lineal2 = LinearRegression()
modelo_lineal2.fit(df_mpg[['weight', 'displacement']], df_mpg['mpg'])

print("R2: "+str(modelo_lineal2.score(df_mpg[['weight', 'displacement']], df_mpg['mpg'])))
print("Coeficientes:", modelo_lineal2.coef_)
print("Intercepto:", modelo_lineal2.intercept_)


















