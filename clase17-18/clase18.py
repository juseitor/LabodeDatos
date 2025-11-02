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
    data=df, x="x", y="y", col="dataset", hue="dataset",
    col_wrap=2, palette="muted", ci=None,
    height=4, scatter_kws={"s": 50, "alpha": 1}
)
#%% primer dataset de anscombe

df1 = df[df['dataset'] == "I"]
df1
X1 = df1['x']
Y1 = df1['y']

xbar = np.mean(X1)
ybar = np.mean(Y1)
b1 = sum((X1-xbar)*(Y1-ybar))/sum((X1-xbar)*(X1-xbar))
b0 = ybar - b1*xbar

#%%
X = np.linspace(min(df1['x']), max(df1['x']))
Y = b0 + b1*X

plt.scatter(df1['x'], df1['y'])
plt.plot(X, Y, 'black')
plt.show()
#%%
Ypred = b0+b1*X1
r2_score(Y1, Ypred)

#%% repetir con los demás datasets de Anscombe



















#%% Autos

df_mpg = pd.read_csv("/home/Estudiante/LabodeDatos/clase17-18/archivosclase17-18/auto-mpg.xls")

X = df_mpg[['weight', 'displacement', 'acceleration']]
y = df_mpg['mpg']

# división en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
#%%

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




















#%%/home/Estudiante


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#%%

df = pd.read_csv('~/LabodeDatos/clase17-18/archivosclase17-18/2025C2 - Alturas - Hoja 1.csv')
df = df.drop(columns = ['¿Sabes quien mas mide 156?', 'Mi mamiiii', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8' ])
df =  df.dropna( subset = ['Altura (cm)', 'Mami'])


#%%
knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(df[['Mami']], df['Altura (cm)'])
ypred = knn.predict(df[['Mami']])
mean_squared_error(df['Altura (cm)'], ypred)
np.sqrt(mean_squared_error(df['Altura (cm)'], ypred))
#%%


