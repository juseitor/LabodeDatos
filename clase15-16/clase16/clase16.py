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






#%% Hago columnas en filas

columnas_titanic = titanic.columns

#%% Agarro solamente el parametro de si sobrevivieron

Y = titanic.Survived

#%%

aributos_titanic = titanic[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Fare']]

#%%

from sklearn.tree import DecisionTreeClassifier as tree

#%%

titanic.dropna(subset = ['Age', 'Fare'], how = 'any', inplace = True)
x = titanic[['Age', 'Fare']]
y = titanic['Survived'].values

titanic.decribe()
arbol_tit = tree.DecisionTreeClassifier(max_depth=3)
arbol_tit.fit(x,y)


#%%

fname1  = '~/LabodeDatos/clase15-16/archivosclase15/arboles.csv'

arboles = pd.read_csv(fname1)

#%%

datos_roundup = pd.read_csv('~/LabodeDatos/clase17-18/archivosclase17-18/datos_roundup.txt', sep = " ")

#%%

a = 106.5 + 0.037*25
b = 106.5 + 0.037*1500

    #%%

#precios = '~/LabodeDatos/preciosbalanza.csv'
#precios_balanza = pd.read_csv(precios)
#
# Si está separado por punto y coma
#df = pd.read_csv(precios, sep=";")
#
# O si sospechás tabulación
#df = pd.read_csv("archivo.csv", sep="\t")
