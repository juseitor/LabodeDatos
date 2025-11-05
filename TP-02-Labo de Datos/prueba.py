import pandas as pd
import numpy as np
import duckdb as db
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
 
#%% Cargamos los dataset

df_minst = pd.read_csv('~/Escritorio/kmnist_classmap_char.csv')
df_kuzu = pd.read_csv('~/Escritorio/kuzushiji_full.csv')

#%%

fig, ax = plt.subplots(figsize = (18,10))

sns.kdeplot(
    data = df_kuzu,
    x = '755',
    fill = False,
    hue = 'label',
    palette = "Paired",
    linewidth = 3,
    ax = ax
    )

ax.set_title("Frecuencia de pixels")
ax.set_xlabel("Pixels")
ax.set_ylabel("Frecuencia de Pixel")

#%%

#
# Clase 0 -> celda 68 al 70 (valor pixel 0)
# Clase 1 -> celda 423 (valor pixel 0)
# 
# Clase 3 -> celda 489 (valor pixel 0)
# Clase 4 -> celda 179
# Clase 5 -> celda 382 (valor pixel 0)
# Clase 6 -> celda 511 (valor pixel 0)
# Clase 7 -> celda 318
#
#
#

#%%

res = []

Class0 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 0
          """
Class0 = db.query(Class0).df()

Dic0 = {}

for i in Class0.iloc[:, :-1].columns:
    Dic0[i] = Class0[i].sum()

Dic0['label'] = 0

res.append(Dic0)


Class1 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 1
          """
Class1 = db.query(Class1).df()

Dic1 = {}

for i in Class1.iloc[:, :-1].columns:
    Dic1[i] = Class1[i].sum()

Dic1['label'] = 1

res.append(Dic1)



Class2 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 2
          """
Class2 = db.query(Class2).df()

Dic2 = {}

for i in Class2.iloc[:, :-1].columns:
    Dic2[i] = Class2[i].sum()

Dic2['label'] = 2
res.append(Dic2)



Class3 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 3
          """
Class3 = db.query(Class3).df()


Dic3 = {}

for i in Class3.iloc[:, :-1].columns:
    Dic3[i] = Class3[i].sum()


Dic3['label'] = 3

res.append(Dic3)
    



Class4 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 4
          """
Class4 = db.query(Class4).df()

Dic4 = {}

for i in Class4.iloc[:, :-1].columns:
    Dic4[i] = Class4[i].sum()


Dic4['label'] = 4
res.append(Dic4)




Class5 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 5
          """
Class5 = db.query(Class5).df()

Dic5 = {}

for i in Class5.iloc[:, :-1].columns:
    Dic5[i] = Class5[i].sum()


Dic5['label'] = 5
res.append(Dic5)




Class6 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 6
          """
Class6 = db.query(Class6).df()

Dic6 = {}

for i in Class6.iloc[:, :-1].columns:
    Dic6[i] = Class6[i].sum()


Dic6['label'] = 6
res.append(Dic6)
  
    


Class7 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 7
          """
Class7 = db.query(Class7).df()

Dic7 = {}

for i in Class7.iloc[:, :-1].columns:
    Dic7[i] = Class7[i].sum()


Dic7['label'] = 7
res.append(Dic7)



Class8 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 8
          """
Class8 = db.query(Class8).df()

Dic8 = {}

for i in Class8.iloc[:, :-1].columns:
    Dic8[i] = Class8[i].sum()


Dic8['label'] = 8
res.append(Dic8)



Class9 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 9
          """
Class9 = db.query(Class9).df()

Dic9 = {}

for i in Class9.iloc[:, :-1].columns:
    Dic9[i] = Class9[i].sum()


Dic9['label'] = 9  
res.append(Dic9)
    
df_final = pd.DataFrame(res)
#%%
#%%  IMAGENES DE LAS LETRAS(LABEL) 

kuzu = df_final.iloc[:, :-1] 


# Plot imagen
img = np.array(kuzu.iloc[4]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()





















#%% Consigna 2

#%% 2.a)

df2 = df_kuzu[(df_kuzu['label'] == 5) | (df_kuzu['label'] == 4)]

#%% 2.b) 

# Separamos en datos de entrenamiento (75% de los mismos), y de test (el restante 25%)
x = df2.drop(columns=['label'])
y = df2['label']

# Notese que estratificamos los casos de train y test para que mantengan la 
#proporción por clase de datos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12, stratify = y)

#%% 2.c)  

# Elegimos probar en algunas series de columnas particulares en las cuales 
#vimos diferencias en el pixel promedio
columnas = [
    [80, 81, 82],
    [110, 111, 112],
    [200, 201, 202],
    [323, 324, 325],
    [425, 426, 427],
    [450, 451, 452],
    [500, 501, 502],
    [550, 551, 552],
    [600, 601, 602],
    [640, 641, 642]
]

# Hacemos Validación cruzada para analizar tambien la estabilidad de haber
#elegido esos atributos (celdas de pixel) en particular
# Notese que usamos StratifiedKFold para que nos mantenga la estratificacion 
#pero ahora para los casos de Validación Cruzada
nsplits = 5
skf = StratifiedKFold(n_splits=nsplits)

# Creamos una matriz de resultados en los cuales su cantidad de columnas j serán
#la prueba hecha en un fold con cada serie de 3 columnas, mientras que las filas i serán
#los 5 folds que elegimos
resultados = np.zeros((nsplits, len(columnas)))



for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
    kf_x_train, kf_x_test = x_train.iloc[train_index], x_train.iloc[test_index]
    kf_y_train, kf_y_test = y_train.iloc[train_index], y_train.iloc[test_index]
    for j in range(len(columnas)):
        columnas_elegidas = columnas[j]
        x_train_columnas = kf_x_train.iloc[:, columnas_elegidas]
        x_test_columnas  = kf_x_test.iloc[:, columnas_elegidas]
        
        clasificador = KNeighborsClassifier(n_neighbors=3)
        clasificador.fit(x_train_columnas, kf_y_train)
        prediccion = clasificador.predict(x_test_columnas)
        accuracy = accuracy_score(kf_y_test, prediccion)
        resultados[i, j] = accuracy

# La matriz resultados que nos queda es el valor de la metrica accuracy en el 
#fold i (0 =< i < 5), en la prueba (cuales columnas) j (0 <= j < 10)
#%%
# Calculamos su promedio y elegimos la serie de columnas con mejor promedio.
resultados_promedio = resultados.mean(axis = 0)

# Nos queda la serie de columnas j = 3 ([323,324,325]) es la que que tuvo mas
#estabilidad en su medida de accuracy. Por lo tanto hacemos calculamos x_test 
#con j = 3
#%%
# Elegimos dentro de nuestros datos de train y test la serie de columnas deseada
x_train_elegido = x_train.iloc[:, columnas[3]]
x_test_elegido = x_test.iloc[:, columnas[3]]

# Entrenamos el modelo con los x_train_elegido y luego evaluamos con el 
#x_test_elegido, para ver su accuracy
clasificador_test = KNeighborsClassifier(n_neighbors = 3)
clasificador_test.fit(x_train_elegido, y_train)
y_pred_test = clasificador_test.predict(x_test_elegido)
accuracy_real = accuracy_score(y_test, y_pred_test)
print(str(accuracy_real))
# Notese que el accuracy_real nos dio un numero muy parecido al promedio de los
#folds hechos con la serie de columnas j = 3.
#%% 2.c) BIEN HECHO
# Elegimos probar en algunas series de columnas particulares en las cuales 
#vimos diferencias en el pixel promedio entre las dos clases
columnas = [
    [80, 81, 82],
    [110, 111, 112],
    [200, 201, 202],
    [323, 324, 325],
    [425, 426, 427],
    [450, 451, 452],
    [500, 501, 502],
    [550, 551, 552],
    [600, 601, 602],
    [640, 641, 642]
]

# Cargamos los datos de resultados en un array en el cual vamos a guardar la
#precision de cada elemento de columnas una vez entrenado el modelo con los 
#datos de train, y testeado con los de test.
resultados = np.zeros(len(columnas))

for i in range(len(columnas)):
    columnas_elegidas = columnas[i]
    x_train_columnas = x_train.iloc[:, columnas_elegidas]
    x_test_columnas  = x_test.iloc[:, columnas_elegidas]
    clasificador = KNeighborsClassifier(n_neighbors=3)
    clasificador.fit(x_train_columnas, y_train)
    prediccion = clasificador.predict(x_test_columnas)
    precision = accuracy_score(y_test, prediccion)
    resultados[i] = precision

# Notese que la mejor precision es la de la serie de columnas 7, que toma los
#valores [550, 551, 552].

#%% 2.c) 
# Volvemos a hacer el mismo procedimiento pero ahora para series de  celdas 
#continuas de 5 elementos. Elegimos una serie continua de columnas[3] ya que 
#es la que nos dio con mejor precisión.












