import pandas as pd
import duckdb as dd

#%%1

EP2 = pd.read_csv("~/LabodeDatos/TP01/Datos_por_departamento_actividad_y_sexo.csv")

#%%2

EP1 = pd.read_csv("~/LabodeDatos/TP01/actividades_establecimientos.csv")

#%%3

POBLACION2022 = pd.read_excel("~/LabodeDatos/TP01/padron_poblacion.xlsX", skiprows=14)

#Elimino primera columna. Sobre la misma variable 

POBLACION2022 = POBLACION2022.drop(POBLACION2022.columns[0], axis=1) 

#%%4

EE2022 = pd.read_excel("~/LabodeDatos/TP01/2022_padron_oficial_establecimientos_educativos.xlsx", skiprows=5)

#%% Consulta Establecimientos Productivos año 2022

cEP2_2022 = """
    SELECT DISTINCT anio, in_departamentos, departamento, provincia_id, provincia, clae6, clae2, letra, genero, Empleo, Establecimientos, empresas_exportadoras
    FROM EP2
    WHERE anio='2022';
    """
            
EP2_2022 = dd.query(cEP2_2022).df()
print(EP2_2022)

#%% Para cada departamento informar la provincia, el nombre del departamento,
#la cantidad de Establecimientos Educativos (EE) de cada nivel educativo,
#considerando solamente la modalidad común, y la cantidad de habitantes con
#edad correspondiente al nivel educativos listado. El orden del reporte debe
#ser alfabético por provincia y dentro de las provincias descendente por
#cantidad de escuelas primarias.

c = """
    SELECT Jurisdiccion, Departamento, 
    FROM EE2022
    """

#%% CONSULTA PARA VER QUE ONDA BENJA1

consultaSQL = """
            SELECT clae6, clae2
            FROM EP1;
            """
            
dataframeResultado = dd.query(consultaSQL).df()#ejecuta la consulta
print(dataframeResultado)


#%% CONSULTA PARA VER QUE ONDA BENJA2

consulta11112 = """
    SELECT DISTINCT clae6, clae2, letra, clae6_desc, clae2_desc, letra_desc
    FROM EP1
    WHERE clae6 = '11112' ;
    """
    
df11112 = dd.query(consulta11112).df()
print(df11112)

#%%