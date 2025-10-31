
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

#%%

df_diabetes = pd.read_csv("~/LabodeDatos/clase15-16/clase15/archivosclase15/diabetes.csv")

#%% X atributos, y etiqueta

X = df_diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df_diabetes['Outcome']

#%% para usar solo los 2 atributos glucosa y bmi

X2 = df_diabetes[['Glucose', 'BMI']].values
y = df_diabetes['Outcome'].values

#%% gráfico de dispersión

plt.figure(figsize=(6, 4))
plt.scatter(X2[:, 0], X2[:, 1], c=y)
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('Distribución de los datos: Glucose vs BMI')
plt.show()

#%% construyo y ajusto el clasificador

clasificador = KNeighborsClassifier(n_neighbors=50)
clasificador.fit(X2, y)

#%% predicción para un nuevo paciente

nuevo_paciente = [[150, 40.0]] 
prediccion = clasificador.predict(nuevo_paciente)
print("Predicción para el nuevo paciente:", "Diabetes" if prediccion[0] == 1 else "No diabetes")








#%% Consigna

from sklearn.datasets import make_moons


#%%

# sklearn.datasets.make_moons(n_samples=100, *, shuffle=True, noise=None, random_state=None)

df_moon = make_moons(n_samples = 200, noise = 0.2)

#%%

x = np.asarray(df_moon[0])
y = np.asarray(df_moon[1])

#%%
df_moon = pd.DataFrame()
#%%
df_moon["continua1"] = x[:, 0]
#%%
df_moon["continua2"] = x[:, 1]
#%%
df_moon["binaria"] = y
#%%                      

fig, ax = plt.subplots()

ax = sns.scatterplot(
    data = df_moon,
    x = "continua1",
    y = "continua2",
    hue =  "binaria"
    )


#%%

clasificador_moon = KNeighborsClassifier(n_neighbors = 2)
clasificador_moon.fit(x,y)

#%%

prueba = [[0.8, -0.15]]
prediccion = clasificador_moon.predict(prueba)
print("La prueba es de variable categórica", "0" if prediccion[0] == 0 else 1)

#%% VAMOS A CALCULAR LA MATRIZ DE CONFUSION Y DE EXACTITUD

from sklearn.metrics import accuracy_score, confusion_matrix

#%% Calculo mis predicciones

y_pred = clasificador_moon.predict(x)

#%% Computo la matriz de confusion comparando con y_pred

confusion_matrix(y,y_pred)

#%% Computo la exactitud comparando con y_pred

accuracy_score(y, y_pred)











#%% Guía de ejercicios

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


#%%

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

#%% 2)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

sns.scatterplot(
    data = data,
    x = "petal length (cm)",
    y = "petal width (cm)",
    hue = "target",
    palette = "Blues",
    edgecolor = 'k',
    ax = ax1
    )

sns.scatterplot(
    data = data,
    x = "sepal length (cm)",
    y = "sepal width (cm)",
    hue = "target",
    palette = "Blues",
    edgecolor = 'k',
    ax = ax2
    )

#ax1.grid(True)
#ax2.grid(True)
ax1.set_title("Scatterplot de Petalo")
ax2.set_title("Scatterplot de Sepalo")

fig.suptitle("Scatterplots de Sepalo y Petalo")



#%% 3)

x_iris = data[['petal length (cm)', 'petal width (cm)']].values
y_iris = data['target']

#%%

clasificador_iris = KNeighborsClassifier(n_neighbors = 3)
clasificador_iris.fit(x_iris, y_iris)

#%%

prueba_iris = [[4.8,1.5]]
prediccion_iris = clasificador_iris.predict(prueba_iris)
print(
    "Predicción para la planta es",
    "0" if prediccion_iris[0] == 0 else
    "1" if prediccion_iris[0] == 1 else
    "2"
)

#%%
y_pred_iris = clasificador_iris.predict(x_iris)
#%%
metrics.confusion_matrix(y_iris,y_pred_iris)
#%%
metrics.accuracy_score(y_iris, y_pred_iris)
