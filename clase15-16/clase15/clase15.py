
import pandas as pd
import matplotlib.pyplot as plt
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

clasificador = KNeighborsClassifier(n_neighbors=10)
clasificador.fit(X2, y)

#%% predicción para un nuevo paciente

nuevo_paciente = [[130, 32.0]] 
prediccion = clasificador.predict(nuevo_paciente)
print("Predicción para el nuevo paciente:", "Diabetes" if prediccion[0] == 1 else "No diabetes")