#%% Limpieza actividades productivas
import pandas as pd
import numpy as np
import pathlib as pl

#Código para ubicar las DB
direccion_actual = pl.Path(__file__).parent.resolve()
str_dir = str(direccion_actual)

EE_df = pd.read_excel(str_dir+'/DB/2022_padron_oficial_establecimientos_educativos.xlsx', 
                         skiprows=6, na_values=' ')
#skiprows saltea las primeras 6 filas que no sirven
#na_values setea que todos los valores str == ' ' a NaN

#%% Limpiar columnas que no nos sirven

# Construyo un DF con las columnas que nos sirven
EE_sin_columnas = EE_df[['Cueanexo', 'Departamento', 'Común', 
                                   'Nivel inicial - Jardín maternal', 
                                   'Nivel inicial - Jardín de infantes',
                                   'Primario', 'Secundario',
                                   'Secundario - INET', 'SNU', 'SNU - INET']]
#Elimino columnas repetidas: https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
EE_sin_columnas = EE_sin_columnas.loc[:,~EE_sin_columnas.columns.duplicated()].copy()

#%% Limpiar filas que no pertenezcan a la modalidad común

EE_sin_columnas['Común'].replace(' ', np.nan, inplace = True)

EE_sin_filas = EE_sin_columnas.dropna(subset = ['Común'])
