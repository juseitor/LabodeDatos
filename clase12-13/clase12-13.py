#%%
import pandas as pd
import duckdb as dd
import matplotlib.pyplot as plt

#%% abrimos y modificamos co2_emissions.csv

fname = '~/LabodeDatos/clase12-13/archivosclase12-13/co2_emissions.csv'
df = pd.read_csv(fname)

df_2022 = df[df["year"]==2022].copy()

#%% 3) pd

grafico3pd= df_2022.plot(
        x = "gdp",
        y = "co2",
        kind="scatter",
        title = "Emisiones de CO2 en 2022", 
#        logy=True
        )

#Funciones de plt
grafico3pd.set_yscale("log")
grafico3pd.set_xscale("log")

#%% 3) plt informal

grafico3plt = plt.scatter(
    df_2022["gdp"],
    df_2022["co2"],
    c = 'r')

plt.xlabel("GDP")
plt.xlabel("CO2")
plt.xscale("log")
plt.yscale("log")
plt.title("Emisiones de CO2 en 2022")
plt.show()

#%% 3) plt formal

#plt.subplots() crea una figura (fig) y uno o más ejes (ax) dentro de ella.
#Es la forma más común y flexible de inicializar un gráfico en Matplotlib
fig, ax = plt.subplots()  #crea la figura y el eje
#fig→ es el objeto “Figure” (la figura completa). 
#Contiene todo el gráfico (título general, varios subgráficos, márgenes, etc.).
#ax → es el objeto “Axes” (el área del gráfico dentro de la figura).
# Técnicamente, Axes es una clase de Matplotlib

ax.scatter(df_2022["gdp"],
           df_2022["co2"],
           c='r')
ax.set_xlabel("GDP")
ax.set_ylabel("CO2")
ax.set_yscale("log")
ax.set_xscale("log")
#fig.suptitle("Emisiones")
ax.set_title("Emisiones de CO2 en 2022") 
#→ Todo eso se hace sobre un objeto Axes.
plt.show()

#ax.set_title() 
#→ cambia el título del gráfico específico (Axes).

#fig.suptitle() 
#→ cambia el título general de la figura completa (Figure), 
#útil cuando hay varios gráficos dentro de una misma figura.

#%% 4) pd

grafico4pd = df_2022.plot(
        x = "population",
        y = "co2",
        kind ="scatter",
        title = "Emisiones de CO2 por Poblacion en 2022" 
        )

#Funciones de plt
grafico4pd.set_yscale("log")
grafico4pd.set_xscale("log")

#%% 4) plt

fig, ax = plt.subplots()  

ax.scatter(df_2022["population"],
           df_2022["co2"],
           c='r')
ax.set_xlabel("Población")
ax.set_ylabel("CO2")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title("Emisiones de CO2 por Poblacion en 2022") 
plt.show()

#%%






#5

def agrupa_otros(df, variable_a_graficar, umbral=0.05):
    """Agrupa en 'Otros' los países cuyo valor es menor a cierto porcentaje del total"""
    df["otrosName"] = df["Name"]
    total = df[variable_a_graficar].sum()
    df.loc[df[variable_a_graficar] / total < umbral, "otrosName"] = "Otros"
    df = df.groupby("otrosName")[[variable_a_graficar]].sum().reset_index()
    return df

df_oil = agrupa_otros(
    df_2022, "oil_co2", umbral=0.03
).copy()  # el nombre agrupado queda en "otrosName"

#%%

graficoCO2 = df_oil.plot(
        x = "otrosName",
        y = "oil_co2",
        kind = "bar",
        title = "Emisiones de CO2 por Paises mas grandes en 2022", 
        legend = True,
        grid = True,
        color = "red",
        )

#%%

#6.1 Repetir lo anterior (con el tipo de gráfico que les parezca mejor) para gas y carbón (coal)

df_coal = agrupa_otros(df_2022,"coal_co2",umbral=0.03).copy()

graficoCOAL_CO2 = df_coal.plot(
        y = "coal_co2",
        labels=df_coal["otrosName"],
        kind = "pie",
        title = "Emisiones de CO2 por Consumo de Carbon en 2022", 
        legend = False,
        grid = True,
        color = "red",
        )
#df_oil.plot(kind="pie", y="oil_co2", labels=df_oil["otrosName"], legend=False)

#%%

df_gas = agrupa_otros(df_2022,"gas_co2",umbral=0.03).copy()

graficoGAS_CO2 = df_gas.plot(
        x = "otrosName",
        y = "gas_co2",
        kind = "bar",
        title = "Emisiones de CO2 por Consumo de Gas en 2022",
        grid = True,
        color = "red",
    )


#%%

#VISUALIZACIONES_2

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#%%

fname1= "~/LabodeDatos/clase12-13/archivosclase12-13/imdb-dataset.csv"

df = pd.read_csv(fname1)

#%%

df["quintiles_duracion"] = pd.qcut(
    df["Duration"], 5, labels=[1, 2, 3, 4, 5]
)  # se le pueden poner otros nombres

#%%








#%% PROBANDO PLT CON WINEDATASET

wine_dataset= "~/LabodeDatos/clase12-13/archivosclase12-13/wine.csv"

wine_dataset = pd.read_csv(wine_dataset, sep = ";")

#print(wine_dataset.columns)

#%%

plt.plot(wine_dataset['alcohol'], wine_dataset['pH'], '.')
#el tercer argumento indica el estilo del marcador(con puntos)

#%%


