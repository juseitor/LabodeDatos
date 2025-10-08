import pandas as pd
import duckdb as dd 

#%%

Carpeta = "~/LabodeDatos/TP01/"

#%%

dato_educativo = pd.read_excel(Carpeta+"2022_padron_oficial_establecimientos_educativos.xlsx",header=6)

#%%

dato_productivo = pd.read_csv(Carpeta+"Datos_por_departamento_actividad_y_sexo.csv")

#%%

claves = pd.read_csv(Carpeta+"actividades_establecimientos.csv")

#%% Elegimos solamente datos que nos competen de EP

Con0 = """
        SELECT departamento, clae6, genero, empleo, Establecimientos
        FROM dato_productivo
        WHERE anio = 2022
        """
productivo = dd.query(Con0).df()

#%% Contamos cuantos EE hay en cada departamento
# CANTIDAD DE EE EN CADA DEPARTAMENTO

Con1 = """
        SELECT Departamento AS departamento, COUNT(*) AS Estable_Educ
        FROM dato_educativo
        GROUP BY Departamento 
        """
        
sum_Estab_Edu = dd.query(Con1).df()

#%% Esto no entendi bien, pero la cuestion es que agrupamos por esos
# atributos y contamos establecimientos

Con2 = """
         SELECT departamento, clae6, genero, Establecimientos
         FROM (
                 SELECT departamento, clae6, genero, Establecimientos,
                 ROW_NUMBER() OVER (
                         PARTITION BY departamento, clae6 
                         ORDER BY Establecimientos DESC) AS rn
                FROM productivo ) 
         WHERE rn = 1
        """
estab_produc = dd.query(Con2).df()

#%% Agarramos de de la ultima consulta y sumamos establecimientos por
# departamento (haciendo mayuscula el atributo departamento)
#CANTIDAD DE ESTABLECIMIENTOS EN CADA DEPARTAMENTO


Con3 =  """
        SELECT UPPER(departamento) AS departamento , SUM(Establecimientos) AS esta_Pro 
        FROM estab_produc
        GROUP BY departamento
        """
sum_estab_produc = dd.query(Con3).df()
        
#%% JUNTAMOS LA SUMA DE EP Y EE

Con4 = """
        SELECT DISTINCT*
        FROM sum_estab_produc
        INNER JOIN sum_Estab_Edu
        ON sum_estab_produc.departamento = sum_Estab_Edu.Departamento
      """
combi = dd.query(Con4).df()

#%% combi tiene dos atributos departamento repetidos. Quitamos uno

Combin = """ 
        SELECT departamento, esta_Pro, estable_Educ
        FROM combi
"""
combinacion = dd.query(Combin).df()

#%% TABLA DE CANTIDAD DE EMPLEADOS POR GENERO EN CADA DEPARTAMENTO

Con6 = """
        SELECT departamento, genero, SUM(Empleo) AS Empelados,
        FROM productivo
        GROUP BY departamento, genero
        """ 
empleados_genero = dd.query(Con6).df() 

#%% CANTIDAD DE EMPLEADOS TOTALES POR DEPARTAMENTO

Con7 = """
        SELECT departamento, SUM(Empleo) AS Empelados,
        FROM productivo
        GROUP BY departamento
        """ 
empleados = dd.query(Con7).df() 
       
#%%