import numpy as np

import pandas as pd

#%%
import random
prueba = random.random()
print(prueba)

import csv
#%%

a = np.array([1,2,3,4,5,6]) #cada lista es una columna
b = np.array ([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) #dos dimensiones
print(a[0])
print(b[0]) #por default es fila
print(b[2][3]) #fila x columna
print(b[2,3])
np.zeros(2) #matriz de ceros del tamaño indicado
np.zeros((2,3))
#%%
import numpy as np

np.arange(4) # array([0,1,2,3])

#%%

import pandas as pd

fname = '/home/jusa/Escritorio/LabodeDatos/clase1-2/archivosclase1/cronograma_sugerido.csv'
df = pd.read_csv(fname)

#%%


