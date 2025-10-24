#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:00:31 2025

@author: mcerdeiro
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris(as_frame = True)

data = iris.frame
X = iris.data
Y = iris.target

iris.target_names
diccionario = dict(zip( [0,1,2], iris.target_names))
#%%

plt.figure(figsize=(10,10))
sns.scatterplot(data = data, x = 'sepal length (cm)' , y =  X['sepal width (cm)'], hue='target', palette='viridis')
plt.savefig('pairplot_iris')
