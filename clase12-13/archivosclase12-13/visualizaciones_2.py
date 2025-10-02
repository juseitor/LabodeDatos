# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# %% 1)
df = pd.read_csv("imdb-dataset.csv")


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

# %% 3)
# Defino pelis polemicas como las del cuartil mas alto de reviews
df.loc[df.cuartil_reviews == 4, "polemica"] = True
df.loc[df.cuartil_reviews != 4, "polemica"] = False

# %% 4) COMPLETAR
sns.histplot(data=df, x="Rating")
plt.title("Histograma de Ratings")

# %% 5) COMPLETAR
sns.kdeplot(data=df, x="metascore_scaled", label="Metascore")
# agregar rating
plt.legend()
plt.title("KDE de Rating y Metascore")


# %% 6) COMPLETAR
media = df[variable].mean()
mediana = df[variable].median()
p16 = df[variable].quantile(0.16)
# calcular percentil 84 y desvío estándar (std)

sns.kdeplot(data=df, x=variable, fill=True)
plt.axvline(media, color="red", linestyle="--", label="Media")

# agregar mediana, p16, p84, mediana - desvío y mediana + desvío

plt.legend()

# %% 7)
sns.kdeplot(data=df, x="Year", fill=True)
# ejes y titulo


# %% 8) COMPLETAR
sns.boxplot(data=df, x="deciles_rating", y="Duration")
# showfliers=False quita los outliers

# %% 9) COMPLETAR
sns.violinplot(data=df, x="decada", y="metascore_scaled")

# %% 10) COMPLETAR
sns.kdeplot(data=df, x="Rating", hue="quintiles_duracion", fill=True)

# %% 11) COMPLETAR

# %% 12) COMPLETAR


# %% 13)
sns.boxplot(
    data=df,
    x="quintiles_metascore",
    y="Duration",
    hue="polemica",
    showfliers=False,
)

# %% 14) COMPLETAR
