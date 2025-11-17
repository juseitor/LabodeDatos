#NOMBRE DE LOS 3 INTEGRANTES 

# Benjamin Francisco Vales

# Gabriel Duham Lopez

# Diego Armando Pariona Escalante 



import pandas as pd
import numpy as np
import pathlib as pl
import duckdb as db
import matplotlib as plt
import seaborn as sns

#%% Código para ubicar las DB

direccion_actual = pl.Path(__file__).parent.resolve()
str_dir = str(direccion_actual) 


#%% Lectura de datos
#%% Establecimientos Educativos

EE_df = pd.read_excel(str_dir+'/TablasOriginales/2022_padron_oficial_establecimientos_educativos.xlsx', 
                         skiprows=6, na_values=' ')
#skiprows saltea las primeras 6 filas, tienen información irrelevante
#na_values setea que todos los valores str == ' ' a nan

#%% Establecimientos Productivos

EP_df = pd.read_csv(str_dir+'/TablasOriginales/Datos_por_departamento_actividad_y_sexo.csv')


#%% GQM 
#%% EP anio
anio = """
        SELECT anio,COUNT(*) AS cant
        FROM EP_df
        WHERE anio = 2021
        GROUP BY anio 
        """
anio = db.query(anio).df()


anio_2021 = anio.loc[0,"cant"]
print(anio_2021)


cant_registros = len(EP_df)
print(cant_registros)

obtenido_anio = anio_2021 / cant_registros
print(obtenido_anio)



solucion_anio = """
                SELECT *
                FROM EP_df
                WHERE anio = 2022
                """
solucion_anio = db.query(solucion_anio).df()


#%% EE Telefono

tel =  """
        SELECT Teléfono , COUNT(*) AS cant
        FROM EE_df
        WHERE Teléfono = '0' OR 
        Teléfono IS NULL
        GROUP BY Teléfono
        """
tel = db.query(tel).df()


telefono = tel.loc[0,"cant"] + tel.loc[1,"cant"]
print(telefono)

obtenido_telefono = telefono / cant_registros
print(obtenido_telefono)


#%%  EE Domiclio


dom =  """
        SELECT Domicilio, COUNT(*) AS cant
        FROM EE_df
        WHERE Domicilio IS NULL
        GROUP BY Domicilio
        """
dom = db.query(dom).df()


domicilio = dom.loc[0,"cant"]
print(domicilio)

obtenido_domicilio = domicilio / cant_registros
print(obtenido_domicilio)


#%% EE Mail


correo =  """
        SELECT Mail , COUNT(*) AS cant
        FROM EE_df
        WHERE Mail IS NULL
        GROUP BY Mail
        """
correo = db.query(correo).df()


mail = correo.loc[0,"cant"]
print(mail)

obtenido_mail = mail / cant_registros
print(obtenido_mail) 

