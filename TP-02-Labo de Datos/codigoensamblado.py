import pandas as pd
import numpy as np
import duckdb as db
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#%%
#%%Código para ubicar las DB

Direccion_actual = pl.Path(__file__).parent.resolve()
Ubi = str(Direccion_actual) 

#%% Cargamos los dataset
df_kuzu = pd.read_csv(Ubi+"/kuzushiji_full.csv")
df_minst = pd.read_csv(Ubi+"/kmnist_classmap_char.csv")


#%%                  EJERCICIO 1 
#%%  
# TODO ESTO PARA CREAR EL DATAFRAME forma_por_clase, NO SE ME OCURRE COMO HACER UN FOR Y REDUCIRLO
# IGUAL SIRVE PARA VER LOS CARACTERES POR CLASE
# ESTAS VARIABLES ESTAN EN CON LA PRIMER LETRA EN MAYUCULA PARA QUE PUEDAD FILTRAR EN EL SPYDER
res = []

#%%
Class0 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 0
          """
Class0 = db.query(Class0).df()

Dic0 = {}

for I in Class0.iloc[:, :-1].columns:
    Dic0[I] = Class0[I].mean() 

Dic0['label'] = 0

res.append(Dic0)

#%%
Class1 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 1
          """
Class1 = db.query(Class1).df()

Dic1 = {}

for I in Class1.iloc[:, :-1].columns:
    Dic1[I] = Class1[I].mean() 

Dic1['label'] = 1

res.append(Dic1)

#%%

Class2 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 2
          """
Class2 = db.query(Class2).df()

Dic2 = {}

for I in Class2.iloc[:, :-1].columns:
    Dic2[I] = Class2[I].mean() 

Dic2['label'] = 2
res.append(Dic2)

#%%

Class3 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 3
          """
Class3 = db.query(Class3).df()


Dic3 = {}

for I in Class3.iloc[:, :-1].columns:
    Dic3[I] = Class3[I].mean() 


Dic3['label'] = 3

res.append(Dic3)

#%%

Class4 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 4
          """
Class4 = db.query(Class4).df()

Dic4 = {}

for I in Class4.iloc[:, :-1].columns:
    Dic4[I] = Class4[I].mean() 


Dic4['label'] = 4
res.append(Dic4)

#%%

Class5 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 5
          """
Class5 = db.query(Class5).df()

Dic5 = {}

for I in Class5.iloc[:, :-1].columns:
    Dic5[I] = Class5[I].mean() 


Dic5['label'] = 5
res.append(Dic5)

#%%

Class6 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 6
          """
Class6 = db.query(Class6).df()

Dic6 = {}

for I in Class6.iloc[:, :-1].columns:
    Dic6[I] = Class6[I].mean() 


Dic6['label'] = 6
res.append(Dic6)

#%%

Class7 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 7
          """
Class7 = db.query(Class7).df()

Dic7 = {}

for I in Class7.iloc[:, :-1].columns:
    Dic7[I] = Class7[I].mean() 


Dic7['label'] = 7
res.append(Dic7)

#%%

Class8 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 8
          """
Class8 = db.query(Class8).df()

Dic8 = {}

for I in Class8.iloc[:, :-1].columns:
    Dic8[I] = Class8[I].mean() 


Dic8['label'] = 8
res.append(Dic8)

#%%

Class9 = """
            SELECT *
            FROM df_kuzu
            WHERE label = 9
          """
Class9 = db.query(Class9).df()

Dic9 = {}

for I in Class9.iloc[:, :-1].columns:
    Dic9[I] = Class9[I].mean() 


Dic9['label'] = 9  
res.append(Dic9)

#%%  

forma_por_clase = pd.DataFrame(res)

#%%
#%%   ELIMINO LA COLUMNA LABEL

kuzu = forma_por_clase.iloc[:, :-1] 

#%%  IMAGENES DE LAS LETRAS(LABEL) 

# Plot imagen
img = np.array(kuzu.iloc[0]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title("forma promedio clase 0")
plt.show()
#%%  un boxplot por clase 

forma_por_clase.T.boxplot()
plt.title("Distribución por tipo")
plt.xlabel("clase")
plt.ylabel("valor maximo por clase")
plt.show()
     
#%%  un grafico de linea por clase 

plt.plot(range(784), forma_por_clase.iloc[7, :-1])
plt.title("claridad por celda promedio de la clase 7")
plt.xlabel("celda de pixeles")
plt.ylabel("valor promedio")
plt.show()


#%%

sns.kdeplot(
    data = df_kuzu,
    x = '500',
    hue = 'label',
    palette = "tab10"
    )








#%% Consigna 2

#%% 2.a)

df2 = df_kuzu[(df_kuzu['label'] == 5) | (df_kuzu['label'] == 4)]

#%% 2.b) 

# Separamos en datos de entrenamiento (80% de los mismos), y de test (el 
#restante 20%)
x = df2.drop(columns=['label'])
y = df2['label']

# Notese que estratificamos los casos de train y test para que mantengan la 
#proporción de datos por clases
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12, stratify = y)

#%% 2.c) 

# Elegimos probar en algunas series de columnas continuas particulares en las 
#cuales vimos diferencias en el pixel promedio entre las dos clases
columnas1 = [
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
resultados_2c_1 = np.zeros(len(columnas1))

for i in range(len(columnas1)):
    columnas_elegidas = columnas1[i]
    x_train_columnas = x_train.iloc[:, columnas_elegidas]
    x_test_columnas  = x_test.iloc[:, columnas_elegidas]
    clasificador = KNeighborsClassifier(n_neighbors=3)
    clasificador.fit(x_train_columnas, y_train)
    prediccion = clasificador.predict(x_test_columnas)
    exactitud = accuracy_score(y_test, prediccion)
    resultados_2c_1[i] = exactitud

# Notese que el mayour valor de exactitud es la de la serie de columnas 7, 
#que toma los valores [550, 551, 552].
#%%
# Volvemos a hacer el mismo procedimiento pero ahora para series de  celdas 
#continuas de 5 elementos. Elegimos continuar con las series de 3 columnas que
#dieron un numero de exactitud mayor a 0.6 en la celda anterior, y reemplazar 
#algunas series de columnas que dieron un numero de exactitud menor a 0.6.

columnas2 = [
    [100,101,102,103,104],
    [125,126,127,128,129],
    [200,201,202,203,204],
    [323,324,325,326,327],
    [425,426,427,428,429],
    [450,451,452,453,454],
    [500,501,502,503,504],
    [550,551,552,553,554],
    [600,601,602,603,604],
    [625,626,627,628,629]
]

resultados_2c_2 = np.zeros(len(columnas2))

for i in range(len(columnas2)):
    columnas_elegidas = columnas2[i]
    x_train_columnas = x_train.iloc[:, columnas_elegidas]
    x_test_columnas  = x_test.iloc[:, columnas_elegidas]
    clasificador = KNeighborsClassifier(n_neighbors=3)
    clasificador.fit(x_train_columnas, y_train)
    prediccion = clasificador.predict(x_test_columnas)
    exactitud = accuracy_score(y_test, prediccion)
    resultados_2c_2[i] = exactitud

# Notese que la mejor precision del anterior ejercicio que la serie de columnas
#7, que toma los valores [550, 551, 552], ahora que le agregamos dos atributos
#continuos mas la precision ([550,551,552,553,554]) nos quedo considerablemente
#menor. Paso de 0.733 a 0.497. Esto sugiere que esta cantidad de atributos 
#elegida es poca para definir el modelo knn, o que son pocos la cantidad de 
#vecinos elegida.
#%% 
# Ahora hacemos una prueba con la continuacion de las series de columnas
#continuas de la celda anterior, pero para series de 10 valores continuos de 
#celdas de pixel (o atributos).

columnas3 = [
    [100,101,102,103,104,105,106,107,108,109],
    [125,126,127,128,129,130,131,132,133,134],
    [200,201,202,203,204,205,206,207,208,209],
    [323,324,325,326,327,328,329,330,331,332],
    [425,426,427,428,429,430,431,432,433,434],
    [450,451,452,453,454,455,456,457,458,459],
    [500,501,502,503,504,505,506,507,508,509],
    [550,551,552,553,554,555,556,557,558,559],
    [600,601,602,603,604,605,606,607,608,609],
    [625,626,627,628,629,630,631,632,633,634]
]

resultados_2c_3 = np.zeros(len(columnas2))

for i in range(len(columnas3)):
    columnas_elegidas = columnas3[i]
    x_train_columnas = x_train.iloc[:, columnas_elegidas]
    x_test_columnas  = x_test.iloc[:, columnas_elegidas]
    clasificador = KNeighborsClassifier(n_neighbors=3)
    clasificador.fit(x_train_columnas, y_train)
    prediccion = clasificador.predict(x_test_columnas)
    exactitud = accuracy_score(y_test, prediccion)
    resultados_2c_3[i] = exactitud

# Notese que por resultados 3 mejoro mucho la exactitud.
#%%
# Por último vamos a repetir nuestro experimento pero continuando con las 
#series de columnas anteriores pero para 15 celdas de pixeles.

columnas4 = [
    [100,101,102,103,104,105,106,107,108,109,110,111,113,114,115],
    [125,126,127,128,129,130,131,132,133,134,135,136,137,138,139],
    [200,201,202,203,204,205,206,207,208,209,210,211,212,213,214],
    [323,324,325,326,327,328,329,330,331,332,333,334,335,336,337],
    [425,426,427,428,429,430,431,432,433,434,435,436,437,438,439],
    [450,451,452,453,454,455,456,457,458,459,460,461,462,463,464],
    [500,501,502,503,504,505,506,507,508,509,510,511,512,513,514],
    [550,551,552,553,554,555,556,557,558,559,560,561,562,563,564],
    [600,601,602,603,604,605,606,607,608,609,610,611,612,613,614],
    [625,626,627,628,629,630,631,632,633,634,635,636,637,638,639]
]

resultados_2c_4 = np.zeros(len(columnas2))

for i in range(len(columnas4)):
    columnas_elegidas = columnas4[i]
    x_train_columnas = x_train.iloc[:, columnas_elegidas]
    x_test_columnas  = x_test.iloc[:, columnas_elegidas]
    clasificador = KNeighborsClassifier(n_neighbors=3)
    clasificador.fit(x_train_columnas, y_train)
    prediccion = clasificador.predict(x_test_columnas)
    exactitud = accuracy_score(y_test, prediccion)
    resultados_2c_4[i] = exactitud

# Notese que hubieron algunos valores de exactitud de resultados_2c_3 los 
#cuales aumentaron con respecto a resultados_2c_3, pero otros disminuyeron.

#%% 2.d)

# Primero vamos a utilizar la ultima lista columnas4 para probar modelos de 
#claisficacion knn para distinta cantidad de vecinos

# Cuantos vecinos vamos a utilizar por cada ciclo
vecinos = [3,5,8,10,15,20,25,50]

# Matriz de resultados. Las columnas j seran los indices de las columnas 
#elegidas de columnas4, mientras que las filas i seran la precision que 
#arroja el modelo con la cantidad de vecinos[i] elegidos.
resultados_2d = np.zeros((len(vecinos), len(columnas4)))

for j in range(len(columnas4)):
    columnas_elegidas = columnas4[j]
    x_train_columnas = x_train.iloc[:, columnas_elegidas]
    x_test_columnas = x_test.iloc[:, columnas_elegidas]
    for i in range(len(vecinos)):
        clasificador = KNeighborsClassifier(n_neighbors = vecinos[i])
        clasificador.fit(x_train_columnas, y_train)
        prediccion = clasificador.predict(x_test_columnas)
        exactitud = accuracy_score(y_test, prediccion)
        resultados_2d[i,j] = exactitud

# Vemos que dentro de una serie continua de columnas de columna4[j], a medida 
#que aumentamos la cantidad de vecinos elegida para algunos casos mejora la 
#metrica exactitud, pero para otros casos empeora ligeramente. Incluso podemos 
#observar que para todas las columnas elegidas salvo un caso (columnas[7]), 
#cuando pasamos el modelo KNN de 25 a 50 vecinos (filas 6 a 7), baja 
#ligeramente el valor de exactitud.
#%%
