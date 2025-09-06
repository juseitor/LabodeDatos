# Evaluacion Clase 3

#%%

import pandas as pd

fname = '/home/Estudiante/Escritorio/LabodeDatos/Clase3/Clase-03-Actividad-01-Datos.csv'
df = pd.read_csv(fname)

#%%

#d = {'lu': ['559/24', '615/23', '156/23'],'nombre':['Pariona', 'Lopez', 'Vales'], 'Transporte': ['Colectivo', 'Colectivo', 'Vehículo individual'], 'aprueba': [True, True, True, False, False, np.nan]}

#df = pd.DataFrame(data = d) # creamos un df a partir de un diccionario
#df.set_index('lu', inplace = True) # seteamos una columna como index

#%%

#Encuesta medios de transporte (Responses) - Form Responses 1.csv
fname = '/home/Estudiante/Escritorio/LabodeDatos/Clase3/Encuesta medios de transporte (Responses) - Form Responses 1.csv'
df = pd.read_csv(fname)



#%%