import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib as pl
import duckdb as db
#%%
#%%Código para ubicar las DB

Direccion_actual = pl.Path(__file__).parent.resolve()
Ubi = str(Direccion_actual) 

#%% Cargamos los dataset
df_minst = pd.read_csv(Ubi+"/kuzushiji_full.csv")
df_kuzu = pd.read_csv(Ubi+"/kmnist_classmap_char.csv")

#%% 
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

#%%

df_final.T.boxplot()
plt.title("Distribución por tipo (Boxplot)")
plt.xlabel("Tipo")
plt.ylabel("Valor")
plt.show()
     








#%% Consigna 2

#%% 2.a)

df2 = df_kuzu[(df_kuzu['label'] == 5) | (df_kuzu['label'] == 4)]

#%% 2.b) 

# Separamos en datos de entrenamiento (75% de los mismos), y de test (el restante 25%)
x = df2.drop(columns=['label'])
y = df2['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12, stratify = y)

#%% 2.c)