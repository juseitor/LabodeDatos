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


#%%                            EJERCICIO 1 
#%%  
# TODO ESTO PARA CREAR EL DATAFRAME forma_por_clase
# ESTAS VARIABLES ESTAN EN CON LA PRIMER LETRA EN MAYUCULA PARA QUE PUEDAD FILTRAR EN EL SPYDER


#%% 
Res = []

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

Res.append(Dic0)

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

Res.append(Dic1)

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
Res.append(Dic2)

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

Res.append(Dic3)

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
Res.append(Dic4)

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
Res.append(Dic5)

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
Res.append(Dic6)

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
Res.append(Dic7)

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
Res.append(Dic8)

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
Res.append(Dic9)

#%%      DATAFRAME FORMAS PROMEDIO POR CLASES 

forma_por_clase = pd.DataFrame(Res)

#%%  
# ELIMINO LA COLUMNA LABEL (CLASE)
kuzu = forma_por_clase.iloc[:, :-1] 

#%%              GRAFICOS SOBRE LAS 10 CLASES    
#%% PODEMOS VER SI LA DISTRIBUCION EN CANTIDAD DE PIXELES ES PARECIDA/SIMILAR 

clases_transpuesta = kuzu.T 

clases_transpuesta.boxplot()
plt.title("Distribución de pixeles por clase")
plt.xlabel("clase")
plt.ylabel("valor maximo por clase")
plt.show()

#%% GRAFICO DE LINEA POR CLASE

for K in range(len(kuzu)):
    plt.plot(range(784), kuzu.iloc[K])
    plt.title("Cantidad promedio de pixeles por celda - clase "+str(K))
    plt.xlabel("celda de pixeles")
    plt.ylabel("valor promedio")
    plt.show()


#%%  HISTOGRAMA POR CLASE

for I in range(len(kuzu)):
    
    GR_CLASES = kuzu.iloc[I]
    GR_CLASES = GR_CLASES.tolist()
    GR_CLASES = np.array(GR_CLASES)



    plt.figure()
    plt.hist(GR_CLASES, bins=20,edgecolor="black")
    plt.title(" Histograma CLASE "+str(I))
    plt.xlabel("Valor de píxel")
    plt.ylabel("Frecuencia de ese valor de pixel")
    plt.grid(alpha=0.15)
    plt.show()
    
#%%  IMAGEN PROMEDIO POR CLASE

for J in range(len(kuzu)):
    # Plot imagen
    img = np.array(kuzu.iloc[J]).reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.title("forma promedio clase "+str(J))
    plt.show()

#%%    B)
#%%   GRAFICOS QUE USAMOS PARA DESCUBRIR LA SIMILITUD ENTRE LA CLASE 1 y 2

#%%  BOXPLOT SOBRE LAS CLASES 1 Y 2

Clases_1_2 = clases_transpuesta[[1,2]]

Clases_1_2.boxplot()
plt.title("Distribución de pixeles - clase 1 Y 2")
plt.xlabel("clase")
plt.ylabel("valor maximo por clase")
plt.show()



#%%  IMAGEN PROMEDIO DE LA CLASE 1 y 2 

# Plot imagen
img = np.array(kuzu.iloc[1]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title("forma promedio clase 1")
plt.show()

# Plot imagen
img = np.array(kuzu.iloc[2]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title("forma promedio clase 2")
plt.show()
#%%  GRAFICO LINEA DE LA CLASE 1 Y 2

plt.plot(range(784), kuzu.iloc[1], label="1")
plt.plot(range(784),kuzu.iloc[2], label="2")
plt.title("Cantidad promedio de pixeles por celda - clase 1 Y 2")
plt.xlabel("celda de pixeles")
plt.ylabel("valor promedio")
plt.legend(title = "CLASE")
plt.grid(alpha = 0.4)
plt.show()

#%% HISTOGRAMA CLASE 1 Y 2 
    
GR_CLASE1 = kuzu.iloc[1]
GR_CLASE1 = GR_CLASE1.tolist()
GR_CLASE1 = np.array(GR_CLASE1)

GR_CLASE2 = kuzu.iloc[2]
GR_CLASE2 = GR_CLASE2.tolist()
GR_CLASE2 = np.array(GR_CLASE2)

       
plt.figure()
plt.hist(GR_CLASE1, bins=20,edgecolor="black", alpha = 0.75, label="1")
plt.hist(GR_CLASE2, bins=20,edgecolor="black", alpha = 0.75, label = "2")
plt.title(" HISTOGRAMA CLASE 1 Y 2" )
plt.xlabel("Valor de píxel")
plt.ylabel("Frecuencia de ese valor de pixel")
plt.legend(title="CLASES")
plt.grid(alpha=0.15)
plt.show()

#%%   GRAFICOS QUE USAMOS PARA COMPARAR LA CLASE 2 Y 6

#%%  BOXPLOT SOBRE LAS CLASES 2 Y 6

Clases_0_7 = clases_transpuesta[[2,6]]

Clases_0_7.boxplot()
plt.title("Distribución de pixeles - clase 2 y 6")
plt.xlabel("clase")
plt.ylabel("valor maximo por clase")
plt.show()


#%%   IMAGEN PROMEDIO CLASE 0

img = np.array(kuzu.iloc[2]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title("forma promedio clase 2")
plt.show()

img = np.array(kuzu.iloc[6]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title("forma promedio clase 6")
plt.show()

#%% GRAFICO DE LINEA CLASE 2 Y 6

plt.plot(range(784), kuzu.iloc[6], label = "6")
plt.plot(range(784), kuzu.iloc[2], label = "2")
plt.title("Cantidad promedio de pixeles por celda - clase 2 Y 6")
plt.xlabel("celda de pixeles")
plt.ylabel("valor promedio")
plt.legend(title = "CLASES")
plt.show()

#%% HISTOGRAMA CLASE 0 Y 7

GR_CLASE6 = kuzu.iloc[6]
GR_CLASE6 = GR_CLASE6.tolist()
GR_CLASE6 = np.array(GR_CLASE6)



plt.figure()
plt.hist(GR_CLASE6, bins=20,edgecolor="black", alpha = 0.75,label = "6")
plt.hist(GR_CLASE2, bins=20,edgecolor="black", alpha = 0.75, label = "2")
plt.title(" HISTOGRAMA CLASE 2 Y 6")
plt.xlabel("Valor de píxel")
plt.ylabel("Frecuencia de ese valor de pixel")
plt.legend(title = "CLASES")
plt.grid(alpha=0.15)
plt.show()
    
#%%  C) 
#%%
# TOMAMOS LA CLASE 8 PARA EL ANALISIS DE ESTE PUNTO
A = [1211,1311,1411]
B = [1511,1609,1611]

#%% IMAGENES DEL GRUPO A 

Caracter1211 = Class8.iloc[:, :-1].iloc[1211]
Caracter1311 = Class8.iloc[:, :-1].iloc[1311]
Caracter1411 = Class8.iloc[:, :-1].iloc[1411]
    
#forma de la letra
Caracter1211 = np.array(Caracter1211).reshape((28,28))
Caracter1311 = np.array(Caracter1311).reshape((28,28))
Caracter1411 = np.array(Caracter1411).reshape((28,28))

fig, A = plt.subplots(1, 3, figsize=(9, 8))

A[0].imshow(Caracter1211 , cmap='gray')
A[1].imshow(Caracter1311 , cmap='gray')
A[2].imshow(Caracter1411 , cmap='gray')

A[0].set_title("CLASE8 - Caracter 1211")
A[1].set_title("CLASE8 - Caracter 1311")
A[2].set_title("CLASE8 - Caracter 1411")

plt.show() 
    
#%% IMAGENES DEL GRUPO B

Caracter1511 = Class8.iloc[:, :-1].iloc[1511]
Caracter1609 = Class8.iloc[:, :-1].iloc[1609]
Caracter1611 = Class8.iloc[:, :-1].iloc[1611]
    
#forma de la letra
Caracter1511 = np.array(Caracter1511).reshape((28,28))
Caracter1609 = np.array(Caracter1609).reshape((28,28))
Caracter1611 = np.array(Caracter1611).reshape((28,28))

fig, B = plt.subplots(1, 3, figsize=(9, 8))

B[0].imshow(Caracter1511 , cmap='gray')
B[1].imshow(Caracter1609 , cmap='gray')
B[2].imshow(Caracter1611 , cmap='gray')

B[0].set_title("CLASE8 - Caracter 1511")
B[1].set_title("CLASE8 - Caracter 1609")
B[2].set_title("CLASE8 - Caracter 1611")

plt.show() 

#%% GRAFICO DE LINEA GRUPO A 
    
plt.plot(range(784), Class8.iloc[:, :-1].iloc[1211], label = "1211")
plt.plot(range(784), Class8.iloc[:, :-1].iloc[1311], label = "1311")
plt.plot(range(784), Class8.iloc[:, :-1].iloc[1411], label = "1411")
plt.title("CLASE 8 - Caracter GRUPO A - valores por celda")
plt.xlabel("celdas de pixeles")
plt.ylabel("valores")
plt.legend(title = "Caracteres")
plt.show()

#%% GRAFICO DE LINEA GRUPO B 
    
plt.plot(range(784), Class8.iloc[:, :-1].iloc[1511], label = "1511")
plt.plot(range(784), Class8.iloc[:, :-1].iloc[1609], label = "1609")
plt.plot(range(784), Class8.iloc[:, :-1].iloc[1611], label = "1611")
plt.title("CLASE 8 - Caracteres GRUPO B - valores por celda")
plt.xlabel("celdas de pixeles")
plt.ylabel("valores")
plt.legend(title = "Caracter")
plt.show()

#%%   Histograma del grupo A

Valores12 = Class8.iloc[:, :-1].iloc[1211].tolist()
Valores13 = Class8.iloc[:, :-1].iloc[1311].tolist()
Valores14 = Class8.iloc[:, :-1].iloc[1411].tolist()

Valores12 = np.array(Valores12)
Valores13 = np.array(Valores13)
Valores14 = np.array(Valores14)
    
Valores12 = Valores12[Valores12 > 0]
Valores13 = Valores13[Valores13 > 0]
Valores14 = Valores14[Valores14 > 0]
    
plt.figure()
plt.hist(Valores14, bins=20,edgecolor="black", alpha = 0.75, label="Carácter 1411")
plt.hist(Valores13, bins=20,edgecolor="black", alpha = 0.75, label="Carácter 1311")
plt.hist(Valores12, bins=20,edgecolor="black", alpha = 0.75, label="Carácter 1211")
plt.title(" Histograma CLASE 8 - Caracteres 1211, 1311 y 1411")
plt.xlabel("Valor de píxel")
plt.ylabel("Frecuencia de ese valor de pixel")
plt.legend(title="Caracter")
plt.grid(alpha=0.15)
plt.show()

#%%   Histograma del grupo B

Valores15 = Class8.iloc[:, :-1].iloc[1511].tolist()
Valores160 = Class8.iloc[:, :-1].iloc[1609].tolist()
Valores161 = Class8.iloc[:, :-1].iloc[1611].tolist()

Valores15 = np.array(Valores15)
Valores160 = np.array(Valores160)
Valores161 = np.array(Valores161)
    
Valores15 = Valores15[Valores15 > 0]
Valores160 = Valores160[Valores160 > 0]
Valores161 = Valores161[Valores161 > 0]
    
plt.figure()
plt.hist(Valores161, bins=20,edgecolor="black", alpha = 0.75, label="Carácter 1611")
plt.hist(Valores15, bins=20,edgecolor="black", alpha = 0.75, label="Carácter 1511")
plt.hist(Valores160, bins=20,edgecolor="black", alpha = 0.75, label="Carácter 1609")
plt.title(" Histograma CLASE 8 - Caracteres 1511, 1609 y 1611 ")
plt.xlabel("Valor de píxel")
plt.ylabel("Frecuencia de ese valor de pixel")
plt.legend(title="Caracteres")
plt.grid(alpha=0.15)
plt.show()



#%% Consigna 2

#%% 2.a)

df2 = df_kuzu[(df_kuzu['label'] == 5) | (df_kuzu['label'] == 4)]
#%%
# Hacemos consulta para saber cuantas letras de Clase 4 hay
CONSULTA_CLASE4 = """
    SELECT COUNT(Label) AS "Cantidad Clase 4"
    FROM df2
    WHERE Label == 4
    GROUP BY Label
"""

consulta_clase4 = db.query(CONSULTA_CLASE4).df()

# Como consulta_clase4 arrojo un valor de 7000 y df2 tiene 14000 columnas, 
#entonces sabemos que hay 7000 filas en df2 para cada clase 

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

# Notese que la mejor exactitud del anterior ejercicio que la serie de columnas
#7, que toma los valores [550, 551, 552], ahora que le agregamos dos atributos
#continuos mas ([550,551,552,553,554]) la exactitud nos quedo considerablemente
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

# Notese que por resultados_2c_3 mejoro mucho la exactitud salvo en un caso,
#el cual es el de columnas3[0]
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

resultados_2c_4 = np.zeros(len(columnas4))

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

#%% 3.a)

# Separamos en datos de entrenamiento (80% de los mismos), y de test (el 
#restante 20%)
X = df_kuzu.drop(columns=['label'])
Y = df_kuzu['label']
#%%
# Notese que nuevamente estratificamos los casos de Desarrollo y Validacion
#para que mantengan la proporción de datos por clases
x_dev, x_eval, y_dev, y_eval = train_test_split(X, Y, test_size=0.2, random_state=12, stratify = Y)

#%% 3.b)

# Al no poder utilizar el conjunto de Held Out que es pedido en el inciso a)
#vamos a utilizar validación cruzada para testear en nuestro casos.

# Notese que usamos StratifiedKFold para que nos mantenga la estratificacion 
#pero ahora para los casos de Validación Cruzada
nsplits = 5
skf = StratifiedKFold(n_splits = nsplits, shuffle = True, random_state = 12)

# Probamos un modelo de arbol para cada profundidad de profundidades
profundidades = [1,2,3,5,8,10]

# Creamos una matriz de resultados en los cuales su cantidad de columnas j serán
#las profundidades del modelo, mientras que las filas i serán los 5 folds que 
#elegimos
resultados_3b = np.zeros((nsplits,len(profundidades)))

for i, (train_index, test_index) in enumerate(skf.split(x_dev, y_dev)):
    skf_x_train, skf_x_test = x_dev.iloc[train_index], x_dev.iloc[test_index]
    skf_y_train, skf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    for j in range(len(profundidades)):
        arbol_3b = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=profundidades[j])
        arbol_3b.fit(skf_x_train, skf_y_train)
        prediccion = arbol_3b.predict(skf_x_test)
        exactitud = accuracy_score(skf_y_test,prediccion)
        resultados_3b[i,j] = exactitud

# Vemos que la exactitud se mantiene en un número similar a lo largo de los 
#folds por cada valor de profundidad del arbol

# Calculamos el promedio de los folds por profundidad del arbol
exactitud_promedio_3b = resultados_3b.mean(axis = 0)

#%% 3.c)

# En este inciso elegimos hacer las pruebas pedidas pero solo para árboles de
#profundidad 10, ya que es el valor de profundidad que arrojo mayor valor de 
#exactitud.

# Creamos nuestro Grid de parametros para el Random Search
parametros_grid = {
    'min_samples_split': [2,5,10,15,20,30],
    'min_samples_leaf': [2,3,5,8,10],
    'max_features': [None, 'sqrt', 'log2']
    }

# Creamos el Árbol de Búsqueda
clasificador_3c = tree.DecisionTreeClassifier(max_depth = 10, criterion = 'entropy')

# Ejecutamos el Random Search. En nuestro caso va a intentar 60 combinaciones 
#de valores de los hiperparámetros elegidos en el Grid
random_search = RandomizedSearchCV(
    estimator = clasificador_3c,
    param_distributions = parametros_grid,
    n_iter = 60,
    cv = 5, #Automaticamente usa StratifiedKFold
    scoring = 'accuracy',
    random_state = 12
    )

# Entrenamos el random_search con el conjunto de desarrollo
random_search.fit(x_dev, y_dev)

# Guardamos en una variable los mejores hiperparámetros probados, y la exactitud
#que arrojó esa combinación de hiperparámetros para el Árbol de profundidad 10
mejores_hiperparametros = random_search.best_params_
exactitud_3c = random_search.best_score_
