#%%
import pandas as pd
import duckdb as dd

#%%

fname = '~/LabodeDatos/clase12-13/archivosclase12-13/co2_emissions.csv'
df = pd.read_csv(fname)

df_2022 = df[df["year"]==2022].copy()

graficoco2= df_2022.plot(
        x = "gdp",
        y = "co2",
        kind ="line",
        title = "Emisiones de CO2 en 2022", 
#        logy=True
        )
graficoco2.set_yscale("log")

#%%

graficopersonasCO2 = df_2022.plot(
        x = "population",
        y = "co2",
        kind ="line",
        title = "Emisiones de CO2 por Poblacion en 2022", 
        )
#graficoco2.set_yscale("log")

#%%

graficopersonasCO2 = df_2022.plot(
        x = "co2",
        y = "population",
        kind ="line",
        title = "Emisiones de CO2 por Poblacion en 2022", 
        )

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