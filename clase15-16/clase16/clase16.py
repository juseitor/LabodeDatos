import pandas as pd


#%%

fname = '~/LabodeDatos/clase15-16/archivosclase16/titanic.csv'
titanic = pd.read_csv(fname)

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
