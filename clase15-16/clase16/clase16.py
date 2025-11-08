import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%

fname = '~/LabodeDatos/clase15-16/clase16/archivosclase16/titanic.csv'
titanic = pd.read_csv(fname)

#%% Consigna 1

fig, ax = plt.subplots(figsize=(8,5))

#En este grafico me muestra el porcentaje de sobrevivientes
#por cada clase.
sns.barplot(
    data = titanic,
    x = 'Pclass',
    y = 'Survived',
    hue = 'Pclass',
#    hue = 'Survived',
    ax = ax
    )
#%%

fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    data = titanic,
    x = 'Sex',
    y = 'Survived',
    hue = 'Sex',
    ax = ax
    )

#%%

titanic["Age Rate"] = pd.cut(
    titanic["Age"],
    bins=[0, 12, 70, float("inf")],
    labels=["0-12", "12-70", "+70"],
    right=True
)
#%%

fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    data = titanic,
    x = 'Age Rate',
    y = 'Survived',
    hue = 'Age Rate',
    ax = ax
    )

#%% Clasificador a mano

def clasificador_titanic(x):
    vive = False
    if x.Sex == "female":
        vive = True
    elif x.Age < 12:
        vive == True
    elif x.Pclass == 1:
        vive == True
    return vive






#%% Parte final de la clase (diapo 50)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

#%% Ahora hacemos lo de clasificacion_titanic.py

df_titanic = pd.read_csv('~/LabodeDatos/clase15-16/clase16/archivosclase16/titanic.csv')

#%% Creamos x e y

#No maneja los valores nan
df_titanic.dropna(subset = ['Age', 'Fare'], how = 'any', inplace = True) 
#inplace=True. Hace el borrado directamente sobre el DataFrame original, sin devolver una copia.
X = df_titanic[['Age', 'Fare']]
y = df_titanic['Survived'].values
#Notese que x es de forma DataFrame

#%% Creamos el arbol

arbol_tit = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
#Tenemos distintos hiperargumentos:
#max_leaf_nodes: Permite limitar directamente la cantidad de hojas y regula el arbol
#min_samples_split: Cantidad mínima de muestras necesarias para intentar dividir un nodo. Default = 2
# min_samples_leaf: Número mínimo de muestras por hoja. Ayuda a controlar sobreajuste aún más que el anterior.
arbol_tit.fit(X,y) #Entrenamiento del modelo

#%%

plt.figure(figsize= [15,10])
tree.plot_tree(
    arbol_tit, #saca la info de aca
    filled = True, 
#Hace que los nodos se coloreen según la clase predominante en cada nodo.
    feature_names = ['Age', 'Fare'], 
#Lista de nombres de las variables predictoras.
    class_names = ['No sobrevive', 'Sobrevive'],
#Son los nombres humanos para cada clase (en vez de 0 y 1)
    rounded = True, 
#Hace que los nodos tengan bordes redondeados, lo que mejora la visualización.
#Si no, son rectángulos rectos.
    fontsize = 15
#Cambia el tamaño del texto dentro de cada nodo del árbol.
#Es muy común aumentarlo porque, por defecto, sale muy chico.
    )

#%% Probamos con distintos valores de x

X = df_titanic[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare',]]

X = pd.get_dummies(X, drop_first=True)
X.columns

y = df_titanic['Survived']

arbol_tit = tree.DecisionTreeClassifier(max_depth=3)
arbol_tit.fit(X, y)

#%%
plt.figure(figsize=[30,10])
tree.plot_tree(arbol_tit, filled=True,
               feature_names=X.columns,
               class_names=['No sobrevive', 'Sobrevive'],
               rounded=True, fontsize=20)
plt.show()

# samples son la cantidad de muestras que llegan a ese nodo
# value signidica cuantas pertenecen a cada clase

#%% Prueba mia

#Primero se hace asi. Como no puede recibir tree variables categóricas,
#entonces con get_dummies las pasamos a discreta

#Pandas detecta todas las columnas categóricas de X (tipo object o category)
#y crea variables dummy solo para esas columnas.
# drop_first = True elimina Sex (la columna original)
X = pd.get_dummies(X, drop_first=True)
X.columns
y = df_titanic['Survived'].values

arbol_titanic_prueba = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
arbol_titanic_prueba.fit(X,y)

plt.figure(figsize=[30,10])
tree.plot_tree(
    arbol_titanic_prueba, 
    filled=True,        
    feature_names = X.columns,        
    class_names=['No sobrevive', 'Sobrevive'],     
    rounded=True, 
    fontsize=20)

plt.show()















#%% Guía de ejercicios

arboles = pd.read_csv('~/LabodeDatos/clase15-16/clase16/archivosclase16/arboles.csv')

#%%

df_jacaranda = arboles[arboles["nombre_com"] == 'Jacarandá']
df_ceibo = arboles[arboles['nombre_com'] == 'Ceibo']
df_pindo = arboles[arboles['nombre_com'] == 'Pindó']
df_eucalipto = arboles[arboles['nombre_com'] == 'Eucalipto']

#%% Grafico de Alturas

fig, ax = plt.subplots()

sns.kdeplot(
    data=df_ceibo,
    x="altura_tot",
    label="Ceibo",
    fill=False,
    color = "k",
    ax=ax
)

sns.kdeplot(
    data=df_jacaranda,
    x="altura_tot",
    label="Jacaranda",
    fill=False,
    color = "r",
    ax=ax
)

sns.kdeplot(
    data=df_pindo,
    x="altura_tot",
    label="pindo",
    fill=False,
    color = "b",
    ax=ax
)

sns.kdeplot(
    data=df_eucalipto,
    x="altura_tot",
    label="Eucalipto",
    fill=False,
    color = "g",
    ax=ax
)

ax.set_xlabel("Altura")
ax.set_ylabel("Frecuencia")
ax.set_title("KDE de alturas por tipo de Arbol")
ax.grid(True)

#%% Grafico de Diametros

fig, ax = plt.subplots()

sns.kdeplot(
    data=df_ceibo,
    x="diametro",
    label="Ceibo",
    fill=False,
    color = "k",
    ax=ax
)

sns.kdeplot(
    data=df_jacaranda,
    x="diametro",
    label="Jacaranda",
    fill=False,
    color = "r",
    ax=ax
)

sns.kdeplot(
    data=df_pindo,
    x="diametro",
    label="pindo",
    fill=False,
    color = "b",
    ax=ax
)

sns.kdeplot(
    data=df_eucalipto,
    x="diametro",
    label="Eucalipto",
    fill=False,
    color = "g",
    ax=ax
)

ax.set_xlabel("Diametro")
ax.set_ylabel("Frecuencia")
ax.set_title("KDE de diametros por tipo de Arbol")
ax.grid(True)
ax.set_xlim(0, 100)
ax.set_xticks(range(0,101,10))

#%% Grafico de Diametros

fig, ax = plt.subplots()

sns.kdeplot(
    data=df_ceibo,
    x="inclinacio",
    label="Ceibo",
    fill=False,
    color = "k",
    ax=ax
)

sns.kdeplot(
    data=df_jacaranda,
    x="inclinacio",
    label="Jacaranda",
    fill=False,
    color = "r",
    ax=ax
)

sns.kdeplot(
    data=df_pindo,
    x="inclinacio",
    label="pindo",
    fill=False,
    color = "b",
    ax=ax
)

sns.kdeplot(
    data=df_eucalipto,
    x="inclinacio",
    label="Eucalipto",
    fill=False,
    color = "g",
    ax=ax
)

ax.set_xlabel("Inclinacion")
ax.set_ylabel("Frecuencia")
ax.set_title("KDE de Inclinacion por tipo de Arbol")
ax.grid(True)
ax.set_xlim(-5, 5)
ax.set_xticks(range(-5,5,1))

#%% 2)

fig, ax = plt.subplots()

sns.scatterplot(
    data = arboles,
    x = 'diametro',
    y = 'altura_tot',
    hue = 'nombre_com',
    ax= ax
    )
ax.set_xlabel("Diametro")
ax.set_ylabel("Altura")
ax.set_title("Scatterplot de Diametro por Altura de Árboles")
ax.grid(True)
#ax.set_xlim(-5, 5)
#ax.set_xticks(range(-5,5,1))

#%%

fig, ax = plt.subplots()

sns.scatterplot(
    data = arboles,
    x = 'inclinacio',
    y = 'altura_tot',
    hue = 'nombre_com',
    ax= ax
    )

ax.set_xlabel("Inclinación")
ax.set_ylabel("Altura")
ax.set_title("Scatterplot de Inclinación por Altura de Árboles")
ax.grid(True)

#%%

from sklearn import  tree

X = arboles[['altura_tot', 'diametro', 'inclinacio']]
y = arboles['nombre_com'].values

arbol_arboles = tree.DecisionTreeClassifier(
    criterion = 'gini', 
    max_depth=3
    )
#%%
#Esto para saber el órden exacto en el que tengo que codear
# el codigo del arbol
print(arbol_arboles.classes_)

#%%
#Tenemos distintos argumentos:
#max_leaf_nodes: Permite limitar directamente la cantidad de hojas y regula el arbol
#min_samples_split: Cantidad mínima de muestras necesarias para intentar dividir un nodo. Default = 2
# min_samples_leaf: Número mínimo de muestras por hoja. Ayuda a controlar sobreajuste aún más que el anterior.

arbol_arboles.fit(X,y) #Entrenamiento del modelo

fig, ax = plt.subplots(figsize = (40,15))

ax = tree.plot_tree(
    arbol_arboles,
    filled = True,
    feature_names = X.columns,
    class_names=['Ceibo', 'Eucalipto', 'Jacarandá', 'Pindó'],
    fontsize = 20,
    rounded = True
    )

#%%

prediccion = arbol_arboles.predict([['22', '56', '8']])

print("Tu predicción para el árbol es",
      "Jacarandá" if prediccion[0] == 'Jacarandá' else
      "Ceibo" if prediccion[0] == 'Ceibo' else
      "Eucalipto" if prediccion[0] == 'Eucalipto' else
      "Pindó"     
      )
