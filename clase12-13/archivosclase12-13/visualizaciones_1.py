# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def agrupa_otros(df, variable_a_graficar, umbral=0.05):
    """Agrupa en 'Otros' los países cuyo valor es menor a cierto porcentaje del total"""
    df["otrosName"] = df["Name"]
    total = df[variable_a_graficar].sum()
    df.loc[df[variable_a_graficar] / total < umbral, "otrosName"] = "Otros"
    df = df.groupby("otrosName")[[variable_a_graficar]].sum().reset_index()
    return df


# %% 1: Abrir dataset

df = ...

# %%### 2: Corte transversal para 2022
df_2022 = ...

# %%### 3:
# Sintaxis general con pandas: df.plot(x="columna_x", y="columna_y", kind="tipo_de_grafico")
# (obviamente, ciertos graficos necesitan otros parametros)

df_2022.plot(x="gdp", y="co2", kind="scatter")

# %%  Repetir pasando ejes a log
# ponerlo lindo (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html)


# %%#### 4: idem para poblacion

# %%#### 5
df_oil = agrupa_otros(
    df_2022, "oil_co2", umbral=0.03
).copy()  # el nombre agrupado queda en "otrosName"
# %%#### 6
# Gráfico de torta:
df_oil.plot(kind="pie", y="oil_co2", labels=df_oil["otrosName"], legend=False)
# ¿que otro tipo de gráfico se les ocurre para ver lo mismo?


# %% 7: elegir uno y repetir para coal y gas


# %%## 8: Pasamos a corte longitudinal, para Argentina (hacer copia)
df_arg = ...

# %%## 9: Generamos columnas per capita
for col in ["co2", "oil_co2", "coal_co2", "gas_co2", "gdp"]:
    df_arg[col + "_per_capita"] = ...

# %%## 10: Graficos de linea juntos
df_arg.plot(
    x="year",
    y=["oil_co2_per_capita", "coal_co2_per_capita", "gas_co2_per_capita"],
    kind="line",
    title="Argentina: CO2 emissions per capita (tons per person)",
)  # los graficos de linea se pueden hacer con varias columnas a la vez

# %%## 11: Calculo porcentajes de emision por tipo de combustible
df_arg["coal_percent"] = df_arg["coal_co2"] / df_arg["co2"]
df_arg["oil_percent"] = df_arg["oil_co2"] / df_arg["co2"]
df_arg["gas_percent"] = df_arg["gas_co2"] / df_arg["co2"]
df_arg["other_percent"] = 1 - (
    df_arg["coal_percent"] + df_arg["oil_percent"] + df_arg["gas_percent"]
)

# %%## a:
df_arg.plot(
    x="year",
    y=["coal_percent", "oil_percent", "gas_percent", "other_percent"],
    kind="bar",
    title="Argentina: CO2 emissions percent by fuel type",
    stacked=True,
)
# Repetir como gráfico de lineas

# %%##
# Hacer un gráfico que muestre si hay correlación entre el gdp per capita y las emisiones de cada tipo per capita
