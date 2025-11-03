import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
#%%

df_minst = pd.read_csv('~/LabodeDatos/TP-02-Labo de Datos/TablasOriginales/kmnist_classmap_char.csv')
df_kuzu = pd.read_csv('~/LabodeDatos/TP-02-Labo de Datos/TablasOriginales/kuzushiji_full.csv')

df_kuzu_5 = df_kuzu[df_kuzu['label'] == 5 ]

#%%

fig, ax = plt.subplots(figsize = (18,10))

sns.kdeplot(
    data = df_kuzu,
    x = '74',
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
df_kuzu_reshaped = df_kuzu.iloc[:,:-1]

img = np.array(df_kuzu_reshaped.iloc[12]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()
