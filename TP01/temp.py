import pandas as pd
import duckdb as dd

#carpeta = "~/LabodeDatos/clase6/archivosclase6/"

#%%

#print(carpeta)

#empleado = pd.read_csv(carpeta+"Datos_por_departamento_actividad_y_sexo.csv")

datos_dep_acti_sex = pd.read_csv("~/LabodeDatos/TP01/Datos_por_departamento_actividad_y_sexo.csv")

acti_establ = pd.read_csv("~/LabodeDatos/TP01/actividades_establecimientos.csv")

#%%

consultaSQL = """
            SELECT clae6, clae2
            FROM acti_establ;
            """
            
dataframeResultado = dd.query(consultaSQL).df()#ejecuta la consulta
print(dataframeResultado)

#%%

consulta2022_departamento = """
    SELECT DISTINCT anio, in_departamentos, departamento, provincia_id, provincia, clae6, clae2, letra, genero, Empleo, Establecimientos, empresas_exportadoras
    FROM datos_dep_acti_sex
    WHERE anio='2022';
    """
            
df1 = dd.query(consulta2022_departamento).df()
print(df1)

#%%