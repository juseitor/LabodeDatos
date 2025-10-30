#%%
import pandas as pd
import duckdb as dd
import seaborn as sns
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

#%% 3) sns

sns.scatterplot(
    data = df_2022,
    x = "gdp",
    y = "co2",
#    hue = "Name"
    )

plt.xlabel("GDP")
plt.xlabel("CO2")
plt.xscale("log")
plt.yscale("log")
plt.title("Emisiones de CO2 en 2022")
plt.show()

#Lo siguente surje de la pregunta de que me llama la 
#atencion no usar una variable para identificar al grafico sns
#Si necesitás modificarlo después, podés guardar el Axes
# que devuelve:
#ax = sns.scatterplot(data=df, x='edad', y='ingresos', hue='genero')
#ax.set_title("Relación entre Edad e Ingresos")
#ax.set_xlabel("Edad")
#ax.set_ylabel("Ingresos")
#Ahora ax es un objeto de tipo matplotlib.axes._axes.Axes, 
#el mismo que usarías con Matplotlib “puro”.

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
#Como grafico4pd es de tipo Ax, set_xlabel son metodos
#del del objeto Axes
#Esto se debe a que pd usa plt, pero devuelve un objeto
#de tipo Ax (la variable grafico4pd). También es de 
#Matplotlib, pero ahora accedés al eje directamente.
grafico4pd.set_xlabel("Población")
grafico4pd.set_ylabel("CO2")

#%% 4) sns

sns.scatterplot(
    data = df_2022,
    x = "population",
    y = "co2",
#    hue = "Name"
#    size = 'Name'
    )   

#Acá xlabel e ylabel son de matplotliob
plt.xlabel("Población")
plt.ylabel("CO2")
plt.xscale("log")
plt.yscale("log")
plt.title("Emisiones de CO2 en 2022")
plt.show()

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

#%% 5)

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

#%% 6) pd

grafico6pd = df_oil.plot(
        x = "otrosName",
        y = "oil_co2",
        kind = "barh",
        title = "Emisiones de CO2 por hidrocarburos en 2022", 
        #legend = True,
        grid = True,
        color = "r",
        )

grafico6pd.set_xlabel("Paises")
grafico6pd.set_ylabel("Hidrocarburos")

#%% 6) sns

fig, ax = plt.subplots()  # creás el Axes manualmente

sns.barplot(
    data = df_oil,
    y = "otrosName",
    x = "oil_co2",
#de esta forma genera barras horizontales sns. Caso contrario
#cambiar x cualitativa e y cuantitativa
    hue = "otrosName",
    ax = ax # Esta es la manera de indicarle a sns que
    # use este Axes
    )

ax.set_xlabel("Hidrocarburos")
ax.set_ylabel("Paises")
ax.set_title("Emisiones de CO2 por hidrocarburos en 2022")
ax.grid(True)

#%% 6) plt

fig, ax = plt.subplots()  

ax.barh(df_oil["otrosName"],
           df_oil["oil_co2"],
# c en plt se usa plt.plot o plt.scatter
           color='k')
ax.set_xlabel("Hidrocarburos")
ax.set_ylabel("Paises")
ax.grid(True)
ax.set_title("Emisiones de CO2 por hidrocarburos en 2022") 
plt.show()

#%% 8)

df_arg = df[df['Name'] == 'Argentina']
df_arg = df_arg.copy()

#%% 9)

df_arg.loc[:,"gdp per Capita"] = df_arg["gdp"] / df_arg["population"]
df_arg.loc[:,"co2 per Capita"] = df_arg["co2"] / df_arg["population"]
df_arg.loc[:,"coal_co2 per Capita"] = df_arg["coal_co2"] / df_arg["population"]
df_arg.loc[:,"oil_co2 per Capita"] = df_arg["oil_co2"] / df_arg["population"]
df_arg.loc[:,"gas_co2 per Capita"] = df_arg["gas_co2"] / df_arg["population"]
#Esto lo hago para el punto 11) porque la suma de los tres no da el valor del atributo "co2"
df_arg.loc[:, "% de emision coal_co2"] = df_arg["coal_co2"] * 100 / (df_arg["coal_co2"] + df_arg["gas_co2"] +df_arg["oil_co2"])
df_arg.loc[:, "% de emision oil_co2"] = df_arg["oil_co2"] * 100 / (df_arg["coal_co2"] + df_arg["gas_co2"] +df_arg["oil_co2"])
df_arg.loc[:, "% de emision gas_co2"] = df_arg["gas_co2"] * 100 / (df_arg["coal_co2"] + df_arg["gas_co2"] +df_arg["oil_co2"])

#%% 10) pd

grafico10pd = df_arg.plot(
    x = "year",
    y = ["oil_co2 per Capita", "coal_co2 per Capita", "gas_co2 per Capita"],
    kind = "line",
    title = "Emisiones de Argentina por tipo de emisión per capita",
    grid = True
    )

grafico10pd.set_xlabel("Año")
grafico10pd.set_ylabel("Emisiones")

#%% 10) sns

#Es mejor no utilizarlo para lineplots ya que no puede
#aceptar mas de un argumento para y

#sns.lineplot(
#    data = df_arg,
#    x = "year",
#    y = ["oil_co2 per Capita", "coal_co2 per Capita", "gas_co2 per Capita"],
#    )

#plt.xlabel("Año")
#plt.ylabel("Emisiones")

#%%

fig, ax = plt.subplots()  

#Como se ve, tenemos que hacerlo uno por uno utilizando
# Axes para que se vean las tres lineas. Queda mas comodo
# en pd
ax.plot(df_arg["year"], df_arg["oil_co2 per Capita"], label="Oil")
ax.plot(df_arg["year"], df_arg["coal_co2 per Capita"], label="Coal")
ax.plot(df_arg["year"], df_arg["gas_co2 per Capita"], label="Gas")
ax.set_xlabel("Año")
ax.set_ylabel("Emisiones")
ax.grid(True)
ax.set_title("Emisiones de CO2 por hidrocarburos en 2022") 
plt.show()

#%% 11) pd

grafico11pd = df_arg.plot(
    x = "year",
    y = ["% de emision coal_co2", "% de emision oil_co2", "% de emision gas_co2"],
    kind = "line",
    grid = True
    )

grafico11pd.set_xlabel("Año")
grafico11pd.set_ylabel("% de Emisiones")
grafico11pd. set_ylim(0,100)

#La de carbon se mantiene estable en proporcion durante los años, 

grafico11bispd = df_arg.plot(
    x = "year",
    y = "gdp per Capita",
    kind = "line",
    grid = True
    )

grafico11bispd.set_xlabel("Año")
grafico11bispd.set_ylabel("PBI per Capita")
grafico11bispd.set_title("Evolución PBI per capita en Argentina")








#%%

#VISUALIZACIONES_2

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#%% 1)

fname1= "~/LabodeDatos/clase12-13/archivosclase12-13/imdb-dataset.csv"

df = pd.read_csv(fname1)

# %% 2)

df["quintiles_duracion"] = pd.qcut(
    df["Duration"], 5, labels=[1, 2, 3, 4, 5]
)  # se le pueden poner otros nombres
df["deciles_rating"] = pd.qcut(df["Rating"], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df["quintiles_metascore"] = pd.qcut(
    df["Metascore"],
    5,
    labels=[1, 2, 3, 4, 5],
)
df["cuartil_reviews"] = pd.qcut(df["reviews"], 4, labels=[1, 2, 3, 4])

#%% 3)

# Defino pelis polemicas como las del cuartil mas alto de reviews
df.loc[df.cuartil_reviews == 4, "polemica"] = True
df.loc[df.cuartil_reviews != 4, "polemica"] = False

#%% 4.1) a) pd

# Esta opcion nos permite no dejar de lado los outliers

grafico44pd = df["Duration"].plot(
    kind = "hist",
    density = True,
    range = (45,746),
    bins = 75,
    logx = True
#    rwidth = 0.1
# bins = Número de intervalos (o una lista con los límites)bins=[0,10,20,30]
    )

grafico44pd.set_xlabel("Duración en minutos")
grafico44pd.set_ylabel("Frecuencia")
grafico44pd.set_title("Frecuencia películas por duración en minutos")
# Esta funcion la tuve que hacer de esta forma porque 
# al estar en escala logaritmica no acepta primer argumento 0
grafico44pd.set_xticks(range(45, 500, 50))

#%% 4.2) a) pd

# La siguiente nos permite poner un rango mas comun,
# pero quitando outliers
grafico444pd = df["Duration"].plot(
    kind="hist",
    bins=75,
    density=True,
    range=(45, 250),
    color="skyblue",
    edgecolor="black",
    alpha=0.7,
    title="Distribución de duración de películas"
)

grafico444pd.set_xlabel("Duración en minutos")
grafico444pd.set_ylabel("Frecuencia")
grafico444pd.set_title("Frecuencia películas por duración en minutos")
# Esta funcion permite agregar ticks cada cierto periodo
grafico444pd.set_xticks(range(0, 250, 30))

#%% 4.1) a) sns

#Tengo que modificar el df porque 

fig, ax = plt.subplots()  # creás el Axes manualmente

sns.histplot(
    data = df,
    x = "Duration",
#    hue = "Year",
    bins = 75,
    binrange = (45,250),
#    binwidth = 10,
    stat = 'density', #'count', 'frequency', o 'probability'
#    kde = True,
# multiple: Si hay hue, cómo combinar barras: 
#'layer', 'stack', 'dodge'
# element
# Estilo de las barras: 'bars', 'step', 'poly'
# log_scale = True
    ax = ax # Esta es la manera de indicarle a sns que
    # use este Axes
    )

ax.set_xlabel("Duración en minutos")
ax.set_ylabel("Frecuencia")
ax.set_title("Frecuencia películas por duración en minutos")
ax.grid(True)
ax.set_xticks(range(0, 250, 30))

#%% 4.2) a) sns

# Si quiero tener el argumento kde = True, por default
# se me hace sobre todos los datos, asi que pierde
# validez el argumento binrange. Si quiero tener el
#primero, entonces tengo que modificarlo anteriormente.

df_subset = df[(df["Duration"] >= 45) & (df["Duration"] <= 250)]

fig, ax = plt.subplots()

sns.histplot(
    data=df_subset,
    x="Duration",
    bins=75,
    kde=True,
    stat = 'density',
    ax=ax
)

ax.set_xlabel("Duración en minutos")
ax.set_ylabel("Frecuencia")
ax.set_title("Frecuencia películas por duración en minutos")
ax.grid(True)
ax.set_xticks(range(45, 251, 30))

#%% 4) a)

fig, ax = plt.subplots()

# Histograma en rango 45 a 250
ax.hist(
    df["Duration"],          # columna de datos
    bins=75,                 # cantidad de bins
    range=(45, 250),         # rango de valores a mostrar
    density=True,            # normaliza para densidad
    color='skyblue',         # color opcional
    edgecolor='black'        # bordes para distinguir los bins
)

# Etiquetas y título
ax.set_xlabel("Duración en minutos")
ax.set_ylabel("Frecuencia")
ax.set_title("Frecuencia de películas por duración en minutos")

# Grilla y ticks
ax.grid(True)
ax.set_xticks(range(45, 251, 30))

plt.show()


#%% 4) b) en sns porque me deja hacer el kde 

#df_subset = df[(df["Rating"] >= 1.3) & (df["Rating"] <= 9.7)]

fig, ax = plt.subplots()

sns.histplot(
    data=df,
    x="Rating",
    bins=85,
    kde=True,
    stat = 'density',
    edgecolor = 'k',
    color = 'b',
    ax=ax
)

ax.set_xlabel("Rating")
ax.set_ylabel("Frecuencia")
ax.set_title("Frecuencia películas por rating")
ax.grid(True)
ax.set_xticks(range(1, 10, 1)) #solo acepta ints
#ax.set_xticks(np.arange(1.3, 9.8, 0.8)) # esta es una buena solucion

#%% 5) a)

# Se puede hacer kde en pd pero muy poco personalizada

fig, ax = plt.subplots()

#Algunos argumentos
#cumulative = True,
#fill=True,
#bw_adjust. Ajusta la suavidad de la curva (valores más altos → más suave)    
#cut. Extiende la curva más allá del rango de los datos
#cumulative. Muestra la distribución acumulada


sns.kdeplot(
    data=df,
    x="Rating",
    label="Rating",
    fill=False,
    color = "b",
    ax=ax
)

sns.kdeplot(
    data=df,
    x="metascore_scaled",
    label="Metascore",
    fill=False,
    color = "r",
    ax=ax
)

ax.set_xlabel("Rating/Metascore")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de ratings")
ax.grid(True)
ax.set_xticks(range(1, 10, 1))

plt.show()

# Se ve que metascore puntea mas facilmente los puntos
# extremos, pero los usuarios como es un promedio entonces
# tiende todo para los 6.5 aprox
# Cambiaria el grafico si en vez del promedio tuviesemos 
# la mediana en la calificacion de los usuarios. O miento, 
# se veria mas o menos el mismo fenómeno

#%% 6) 



#%% 7)

fig, ax = plt.subplots()

sns.kdeplot(
    data = df,
    x = "Year",
    color = "b",
    fill = False,
    ax=ax)

ax.set_xlabel("Años")
ax.set_ylabel("Frecuencia")
ax.set_title("Frecuencia de películas por años")
ax.grid(True)
ax.set_xticks(range(1929,2026,10))

#%% 8) df

fig, ax = plt.subplots(figsize=(8,5)) #ancho y alto en pulgadas

ax = df.boxplot(
    column = "Duration",
    by = "deciles_rating",
    showfliers = False, # elimina outliers
    grid = True,
    patch_artist = True, #rellena las cajas de color
    ax=ax 
    )

ax.set_xlabel("Decil de Rating")
ax.set_ylabel("Duración de la película")
#ax.set_title("")
fig.suptitle("")
ax.set_title("Boxplot de duración de película por decil de rating")

# Veo el patron de que a medida que van aumentando los ratings
#las películas duran mas.

#%% 8) sns

fig, ax = plt.subplots(figsize=(8,5))

sns.boxplot(
    data = df,
    x = "deciles_rating",
    y = "Duration",
    ax = ax,
    showfliers = False,      # elimina outliers
    palette = "Blues",       # paleta de colores
    width = 0.6,             # ancho de las cajas
    linewidth = 1.2,         # grosor de bordes
    boxprops = {"alpha": 0.8}  # transparencia suave
)

# Personalización de ejes y título
ax.set_xlabel("Decil de Rating", fontsize=11)
ax.set_ylabel("Duración de la película (minutos)", fontsize=11)
ax.set_title("Boxplot de duración de películas por decil de rating", fontsize=13, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout() #ajusta automáticamente los espacios y márgenes del gráfico para que los títulos
plt.show()


#%% 9) sns

#No se puede hacer en pd

fig, ax = plt.subplots(figsize=(10,5))

sns.violinplot(
    data=df,
    x="decada",
    y="metascore_scaled",
#    inner="box",       # Muestra el boxplot dentro del violín
    cut=0,             # Evita que se extienda más allá de los datos
    bw=0.2,            # Controla el "suavizado" del KDE
    scale="width",     # Ajusta el ancho de cada violín
    palette="Blues",   # Colores
    ax=ax
)

ax.set_xlabel("Decadas")
ax.set_ylabel("metascore_scaled")
ax.set_title("Decada por Metascore")

plt.tight_layout() #ajusta automáticamente los espacios y márgenes del gráfico para que los títulos
plt.show()

#%% 9) Boxplot de comparacion

fig, ax = plt.subplots(figsize=(10,5))

sns.boxplot(
    data = df,
    x="decada",
    y="metascore_scaled",
    ax = ax,
    showfliers = False,      # elimina outliers
    palette = "Blues",       # paleta de colores
    width = 0.6,             # ancho de las cajas
    linewidth = 1.2,         # grosor de bordes
    boxprops = {"alpha": 0.8}  # transparencia suave
    )

ax.set_xlabel("Decadas")
ax.set_ylabel("metascore_scaled")
ax.set_title("Decada por Metascore")
plt.tight_layout() #ajusta automáticamente los espacios y márgenes del gráfico para que los títulos
plt.show()

#%% 10)

#fig, ax = plt.subplots(figsize=(10,5))

#sns.kdeplot(
#    data = df,
#    )
