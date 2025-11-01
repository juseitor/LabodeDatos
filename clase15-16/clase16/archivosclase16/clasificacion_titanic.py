#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mcerdeiro
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd

#%%######################

df_titanic = pd.read_csv('~/LabodeDatos/clase15-16/clase16/archivosclase16/titanic.csv')

df_titanic.dropna(subset = ['Age', 'Fare'], how = 'any', inplace = True)
X = df_titanic[['Age', 'Fare']]
y = df_titanic['Survived'].values

df_titanic.describe()
arbol_tit = tree.DecisionTreeClassifier(max_depth=3)
arbol_tit.fit(X,y)

#%%
plt.figure(figsize= [15,10])
tree.plot_tree(arbol_tit,filled = True,
               feature_names = ['Age', 'Fare'], class_names = ['No sobrevive', 'Sobrevive'],rounded = True, fontsize = 15)
#%%
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






























