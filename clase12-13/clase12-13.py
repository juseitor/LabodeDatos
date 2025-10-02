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

graficoCO2 = df_2022.plot(
        x = "Name",
        y = "oil_co2",
        title = "Emisiones de CO2 por Paises mas grandes en 2022", 
        legend = True,
        grid = True,
        color = "red",
        )

#%%

#6 Hacer dos gráficos distintos que muestre los países que más CO2 emiten por consumo de petróleo

graficoCO2x = df_2022.plot(
        x = "oil_co2",
        y = "co2",
        title = "Emisiones de CO2 por Consumo por petroleo en 2022", 
        legend = True,
        grid = True,
        color = "red",
        )



#%%