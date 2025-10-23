import pandas as pd
import numpy as np
import pathlib as pl
import duckdb as db
#import matplotlib as plt
#import seaborn as sns

#%%Código para ubicar las DB

direccion_actual = pl.Path(__file__).parent.resolve()
str_dir = str(direccion_actual) 


#%% Lectura de datos
#%% EDUCATIVO

EE_df = pd.read_excel(str_dir+'/TablasOriginales/2022_padron_oficial_establecimientos_educativos.xlsx', 
                         skiprows=6, na_values=' ')
#skiprows saltea las primeras 6 filas, tienen información irrelevante
#na_values setea que todos los valores str == ' ' a nan

#%% PRODUCTIVO

EP_df = pd.read_csv(str_dir+'/TablasOriginales/Datos_por_departamento_actividad_y_sexo.csv')











#%%               GQM 
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















#%% Limpieza establecimientos_educativos
#%% Limpieza columnas

#Construyo un DF con las columnas que nos sirven
EE_limpio = EE_df[['Jurisdicción','Departamento','Común', 'Nivel inicial - Jardín maternal', 
                                   'Nivel inicial - Jardín de infantes',
                                   'Primario', 'Secundario',
                                   'Secundario - INET', 'SNU', 'SNU - INET']]
#Elimino columnas repetidas: https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
EE_limpio = EE_limpio.loc[:,~EE_limpio.columns.duplicated()].copy()

#%% Limpieza de filas que no pertenezcan a la modalidad común

EE_limpio['Común'].replace(' ', np.nan, inplace = True)

#%%

EE_limpio = EE_limpio.dropna(subset = ['Común'])

#%% Descarto la columna 'Común' porque no aporta información.

EE_limpio = EE_limpio.drop(['Común'], axis = 1)

#%%
#Renombro las columnas 'Nivel inicial - Jardín maternal',
#'Nivel inicial - Jardín de infantes', 'Secundario - INET', 'SNU - INET'

EE_limpio = EE_limpio.rename(columns = {'Nivel inicial - Jardín maternal': 'JardinM',
                            'Nivel inicial - Jardín de infantes': 'JardinI',
                            'Secundario - INET': 'SecundarioINET'
                            , 'SNU - INET': 'SNUINET'})

#%% Convierto los nombres de los departamentos a mayuscula 

mayus = """
        SELECT UPPER(Jurisdicción) as Provincia, UPPER(Departamento) AS Departamento, JardinM, 
                                   JardinI,
                                   Primario, Secundario,
                                   SecundarioINET, SNU, SNUINET
        FROM EE_limpio
        """
EE_limpio =  db.query(mayus).df()

#%% Modificar los nombres de Departamento para que coincidan con AP

#limpio lo que son acentos
acentos_departamento =  """
            SELECT Provincia,
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(Departamento, 'Á', 'A'),'É', 'E'),'Í', 'I'),
            'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS Departamento, JardinM, JardinI, Primario, Secundario, SecundarioINET, SNU, SNUINET
            FROM EE_limpio
            
            """
EE_limpio = db.query(acentos_departamento).df()

#%% Modificar los nombres de Provincia para que coincidan con AP

#limpio lo que son acentos
acentos_provincia =  """
            SELECT 
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(Provincia, 'Á', 'A'),'É', 'E'),'Í', 'I'),
            'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS Provincia, Departamento, JardinM, JardinI, Primario, Secundario, SecundarioINET, SNU, SNUINET
            FROM EE_limpio
            
            """
EE_limpio = db.query(acentos_provincia).df()

#%%

# Tomo como referencias los nombre de departamentos y provincias de E. Productivos y edito los de E.Educativos  ----> (lo que tengo, lo que quiero)      
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("1§ DE MAYO", "1° DE MAYO")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("GENERAL ANGEL V PEÑALOZA", "ANGEL VICENTE PEÑALOZA")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("CORONEL DE MARINA L ROSALES", "CORONEL DE MARINA LEONARDO ROSALES")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("CORONEL FELIPE VARELA", "GENERAL FELIPE VARELA")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("DOCTOR MANUEL BELGRANO", "DR. MANUEL BELGRANO")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("GENERAL JUAN F QUIROGA", "GENERAL JUAN FACUNDO QUIROGA")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("GENERAL JUAN MARTIN DE PUEYRREDON", "JUAN MARTIN DE PUEYRREDON")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("GENE Jardin, Pirmaria, Secundaria, Poblacion TotalRAL OCAMPO",  "GENERAL ORTIZ DE OCAMPO")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("JUAN B ALBERDI", "JUAN BAUTISTA ALBERDI")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("JUAN F IBARRA", "JUAN FELIPE IBARRA"  )
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("MAYOR LUIS J FONTANA", "MAYOR LUIS J. FONTANA")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("O HIGGINS", "O'HIGGINS")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("GENERAL OCAMPO" , "GENERAL ORTÍZ DE OCAMPO")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("LIBERTADOR GRL SAN MARTIN" , "LIBERTADOR GENERAL SAN MARTIN")
#EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("O HIGGIN", "O'HIGGINS")
EE_limpio["Provincia"] = EE_limpio["Provincia"].replace("CIUDAD DE BUENOS AIRES", "CABA")










#%% Limpieza Establecimientos_Productivos
#%% Filtro por año = 2022

EP_limpio = EP_df[EP_df['anio'] == 2022]

#%% Esto es para hacer lo de poblacion mas adelante... 

EP_limpioo = EP_limpio[['departamento','provincia_id','provincia', 'clae6', 'genero', 
                       'Empleo', 'Establecimientos', 'empresas_exportadoras']]

#%% Dejo sólo las columnas necesarias

EP_limpio = EP_limpio[['departamento', 'provincia', 'clae6', 'genero', 
                       'Empleo', 'Establecimientos', 'empresas_exportadoras']]


#%% Asigno los nombres de los atributos como en el modelo Relacional, y hago mayusculas
# a los valores de Provincia y Departamento como en EE

consulta_EP = """
        SELECT UPPER(provincia) AS Provincia, 
        UPPER(departamento) AS Departamento,
        clae6 AS Clae6,
        genero as Sexo,
        Empleo AS Empleados,
        Establecimientos,
        empresas_exportadoras AS Empresas_exportadoras
        FROM EP_limpio
"""

EP_limpio = db.query(consulta_EP).df()

#%% Modificamos los nombres de Departamento para que no tengan tildes

#SIGO VIENDO TILDES SI COPIO, PEGO, Y ADAPTO ESTA MISMA FUNCION DE EE

#limpio lo que son acentos
acentos_departamento_EP =  """
            SELECT Provincia,
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(Departamento, 'Á', 'A'),'É', 'E'),'Í', 'I'),
            'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS Departamento, Clae6, Sexo, Empleados, Establecimientos, Empresas_exportadoras
            FROM EP_limpio
            
            """
EP_limpio = db.query(acentos_departamento_EP).df()

#%% Modificar los nombres de Provincia para que coincidan con AP

#SIN EMBARGO SI FUNCIONA ACA

#limpio lo que son acentos
acentos_provincia_EP =  """
            SELECT 
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(Provincia, 'Á', 'A'),'É', 'E'),'Í', 'I'),
            'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS Provincia, Departamento,
            Clae6, Sexo, Empleados, Establecimientos, Empresas_exportadoras
            FROM EP_limpio
            
            """
EP_limpio = db.query(acentos_provincia_EP).df()









#%% Limpieza pardón_población 
#%%

#abro el excel
poblacion = pd.read_excel(str_dir+"/TablasOriginales/padron_poblacion.xlsX")

#%%

#elimino la primera columna que no me sirve 
poblacion = poblacion.iloc[:, 1:]

#%%

#elimino la ultima tabla que es el resumen que tampoco me servira
poblacion = poblacion.iloc[:56596]

#%%  Obtengo los indices, los nombres de los departamentos y su codigo de la provincia a la que corresponden
#%%

departamentos = []
codigos = []
indices=[]


#primero recorro todas las filas
for i in range(len(poblacion)):
    valor = str(poblacion.iloc[i, 0]) #obtengo lo que hay en la primera columna
    if "AREA #" in valor: # si es que esta el patron AREA #... entonces
        indices.append(i)  # me guardo el indice
        departamentos.append(poblacion.iloc[i, 1]) # me guardo el departamento que esta en la segunda columna
       
        codigo = poblacion.iloc[i,0].split()[-1] 
        codigos.append(int(codigo)//1000) # me guardo el id de la provincia que es el primero o los primeros dos digitos


#%% teniendo los indices recorro por tablas y me guardo los datos 
#%%
resultados = []

#por cada aparicion de AREA #... me separo su tabla correspondiente
for i in range(len(indices)):
    #marco el final de la tabla como el inicio de una nueva(que es donde aparece AREA...)
    if i < len(indices) - 1:
        final = indices[i + 1]
        #Y si estoy en la ultima tabla marco como el final el final del excel(ya que elimine el resumen)
    else:
        final = len(poblacion)
    
    inicio = indices[i]
    #Aca me separo el bloque con el que trabajare
    #le agrego el .reset_index(drop=True) para que bloque sea un dataframe independiente al de poblacion
    # y no tener problemas de copia o estar modificando el dataframe original cuando no quiero eso
    bloque = poblacion.loc[inicio+1:final-1].reset_index(drop=True) 
    
    
    depto = departamentos[i] 
    depto = str(depto)
    id_provincia = codigos[i]
    
    
    #Las columnas edades y casos las convierto en int, porque puede ser que se vea como numeros pero son str(que es lo que pasa)
    bloque["Unnamed: 1"] = pd.to_numeric(bloque["Unnamed: 1"], errors='coerce')
    bloque["Unnamed: 2"] = pd.to_numeric(bloque["Unnamed: 2"], errors='coerce')
    
    
    # Elimino filas que tienen como valor nulls, por ej arriba converti la fila con el total en nulls entonces aca la estoy eliminando y solo me quedo con las edades validas
    bloque = bloque.dropna(subset=["Unnamed: 1", "Unnamed: 2"])
    

    #Aca filtro el dataframe BLOQUE, que seria la tabla del departamento x, donde me quedo con las filas que cumplen el rango que yo quiero y las sumo
    jardin = bloque[(bloque["Unnamed: 1"] >= 3) & (bloque["Unnamed: 1"] <= 5)]["Unnamed: 2"].sum()
    primaria = bloque[(bloque["Unnamed: 1"] >= 6) & (bloque["Unnamed: 1"] <= 12)]["Unnamed: 2"].sum()
    secundaria = bloque[(bloque["Unnamed: 1"] >= 13) & (bloque["Unnamed: 1"] <= 18)]["Unnamed: 2"].sum()
    
    #Aca directamente sumo todas las filas para obtener el total
    total = bloque["Unnamed: 2"].sum()
    

    #Con los datos obtenidos me creo un diccionario, los agrego a la lista ya creada antes"
    resultados.append(
         {"id_Provincia": id_provincia,
          "Departamento": depto,
          "Jardin": jardin,
          "Primaria": primaria,
          "Secundaria": secundaria,
          "Poblacion Total": total}
        )

#al finalizar el for obtengo una lista de diccionarios, solo queda convertirlo a DataFrame
df_final = pd.DataFrame(resultados)















#%% Modelado de DB
#%% Creamos Departamento

crearDepartamento = """
                    SELECT DISTINCT Departamento AS Nombre, Provincia
                    FROM EP_limpio;
                    """

Departamento = db.query(crearDepartamento).df()

#%% Esta consulta permite que en vez de ordenar solo por departamento y 
# se nos computen dos departamentos distintos con el mismo nombre como
# el mismo, hace que Provincia y Departamento en conjunto sean la clave.

crearEE = """
    SELECT Provincia, Departamento,
    SUM(CASE WHEN JardinM = 1 OR JardinI = 1 THEN 1 ELSE 0 END) AS Cantidad_Jardines,
    SUM(CASE WHEN Primario = 1 THEN 1 ELSE 0 END) AS Cantidad_Primarios,
    SUM(CASE WHEN Secundario = 1 OR SecundarioINET = 1 OR SNU = 1 OR SNUINET = 1 THEN 1 ELSE 0 END) AS Cantidad_Secundarios,
    COUNT(*) AS Cantidad_EE
FROM EE_limpio
GROUP BY Provincia, Departamento
ORDER BY Provincia, Departamento;
"""
Establecimientos_Educativos = db.query(crearEE).df()

#%% CONSULTA PERSONAL Esta es una consulta para corroborar que EP y EE
#tienen la misma cantidad de Departamentos

CONSULTA_PERSONAL = """
                    SELECT DISTINCT Departamento AS Nombre, Provincia
                    FROM Establecimientos_Educativos;
                    """

Departamento_personal = db.query(CONSULTA_PERSONAL).df()

#%% Asignmanos la variable EP_limpio a Establecimientos_Productivos porque 
# ya esta limpiado y acorde al Modelo Relacional

Establecimientos_Productivos =  EP_limpio

#%% AGREGO LA COLUMNA Provincia
#%%
pro =   """
        SELECT DISTINCT UPPER(departamento) AS departamento, provincia_id, UPPER(provincia) AS provincia
        FROM EP_limpioo
        """
pro = db.query(pro).df()

prod = """
        SELECT
        REPLACE(
        REPLACE(
        REPLACE(
        REPLACE(
        REPLACE(
        REPLACE(departamento, 'Á', 'A'),'É', 'E'),'Í', 'I'),
        'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS departamento,
        provincia_id, provincia
        FROM pro
        """
prod = db.query(prod).df()



po = """
        SELECT id_Provincia, UPPER(Departamento) AS Departamento, Jardin AS Población_jardin,
        Primaria AS Población_primario, Secundaria AS Población_secundario,
        "Poblacion Total" AS Cantidad_habitantes
        FROM df_final
"""
po = db.query(po).df()


pob = """
        SELECT id_Provincia,
        REPLACE(
        REPLACE(
        REPLACE(
        REPLACE(
        REPLACE(
        REPLACE(Departamento, 'Á', 'A'),'É', 'E'),'Í', 'I'),
        'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS Departamento,
        Población_jardin, Población_primario, Población_secundario, Cantidad_habitantes
        FROM po
        """
pob = db.query(pob).df()

pob.iat[235,1] = "1° DE MAYO"

Pobla = """
                SELECT DISTINCT *
                FROM pob
                LEFT OUTER JOIN prod
                ON pob.Id_Provincia = prod. provincia_id AND 
                prod.departamento = pob.Departamento
                """
Pobla = db.query(Pobla).df()




poblac =      """SELECT provincia AS Provincia, Departamento, Población_jardin,
                 Población_primario, Población_secundario, Cantidad_habitantes
                 FROM Pobla
                        """
Población = db.query(poblac).df()

























#%%CONSULTAS SQL
#%% 1) SQL

SQL1 = """
    SELECT ee.Provincia, ee.Departamento, 
    ee.Cantidad_Jardines AS Jardines, 
    p.Población_jardin AS "Población Jardin", 
    ee.Cantidad_Primarios AS Primarias, 
    p.Población_primario AS "Población Primaria", 
    ee.Cantidad_Secundarios AS Secundarios, 
    p.Población_secundario AS "Población Secundaria"
    FROM Establecimientos_Educativos AS ee
    INNER JOIN Población AS p
    ON ee.Provincia = p.Provincia AND ee.Departamento = p.Departamento
    ORDER BY ee.Provincia ASC, Primarias DESC;
"""

sql1 = db.query(SQL1).df()

#%% 2) SQL

SQL2 = """
    SELECT Provincia, Departamento, 
    SUM (Empleados) AS "Cantidad total de empleados en 2022" 
    FROM Establecimientos_Productivos
    GROUP BY Provincia, Departamento
    ORDER BY Provincia ASC, "Cantidad total de empleados en 2022" DESC
"""

sql2 = db.query(SQL2).df()
#%% ESTO ES PARA BORRAR

#sql2.to_csv('~/LabodeDatos/TP01/ConsultaSQL2.csv', index = False)

#%% 3) SQL

SQL3 = """
    SELECT  DISTINCT ee.Provincia, ee.Departamento, 
    SUM(CASE WHEN ep.Sexo = 'Mujeres' THEN ep.Empresas_exportadoras ELSE 0 END) AS Cant_Expo_Mujeres, 
    ee.Cantidad_EE AS Cant_EE, 
    p.Cantidad_habitantes AS Población
    FROM Establecimientos_Educativos AS ee
    INNER JOIN Establecimientos_Productivos AS ep
    ON ee.Provincia = ep.Provincia AND ee.Departamento = ep.Departamento
    INNER JOIN Población as p
    ON ee.Provincia = p.Provincia AND ee.Departamento = p.Departamento   
    GROUP BY ee.Provincia, ee.Departamento, ee.Cantidad_EE, p.Cantidad_habitantes
    ORDER BY Cant_EE DESC, Cant_Expo_Mujeres DESC, ee.Provincia ASC, ee.Departamento ASC;
"""

sql3 = db.query(SQL3).df()

#%% 4) SQL

# Consultamos la cantidad de empleo por departamento
CANTIDAD_EMPLEOS_X_DEPARTAMENTO = """
    SELECT Provincia, Departamento, SUM(Empleados) AS "Cant. empleos"
    FROM Establecimientos_Productivos
    GROUP BY Provincia, Departamento;
"""

cantidad_empleos_x_departamento = db.query(CANTIDAD_EMPLEOS_X_DEPARTAMENTO).df()

#%%

# Seleccionamos los departamentos que por cada provincia tienen mayor 
# cantidad de empleos promedio
PROMEDIO = """
    SELECT Provincia, Departamento, "Cant. empleos"
    FROM cantidad_empleos_x_departamento AS c1
    WHERE "Cant. empleos" > (
        SELECT AVG("Cant. empleos")
        FROM cantidad_empleos_x_departamento AS c2
        WHERE c2.Provincia = c1.Provincia)
    ORDER BY Provincia, "Cant. empleos" DESC;
"""

promedio = db.query(PROMEDIO).df()

#%% 

#Uno Establecimientos_Productivos con promedio para que me queden los
#departamentos con cantidad de empleados mayor al promedio

LIMPIEZA_EP_MAYOR_PROMEDIO = """
    SELECT pro.Provincia, pro.Departamento, ep.Clae6, ep.Sexo, ep.Empleados
    FROM Establecimientos_Productivos AS ep
    INNER JOIN promedio as pro
    ON ep.Provincia = pro.Provincia AND ep.Departamento = pro.Departamento;
"""
limpieza_EP_mayor_promedio = db.query(LIMPIEZA_EP_MAYOR_PROMEDIO).df()

#%%

# Agrupamos hombres y mujeres en "Cant. empleos" por clae por departamento

AGRUPAMOS_X_SEXO = """
    SELECT Provincia, Departamento, Clae6, SUM(Empleados) AS "Cant. empleos"
    FROM limpieza_EP_mayor_promedio
    GROUP BY Provincia, Departamento, Clae6
"""

agrupamos_x_sexo = db.query(AGRUPAMOS_X_SEXO).df()

#%%

#Ahora nos quedamos con la actividad que mas empleo genere por cada Departamento

QUEDAMOS_CON_MAX = """
    SELECT a.Provincia, a.Departamento, a.Clae6, a."Cant. empleos"
    FROM agrupamos_x_sexo AS a
    INNER JOIN (
        SELECT Provincia, Departamento, MAX("Cant. empleos") AS max_empleos
        FROM agrupamos_x_sexo
        GROUP BY Provincia, Departamento) AS b
    ON a.Provincia = b.Provincia AND a.Departamento = b.Departamento AND a."Cant. empleos" = b.max_empleos
    ORDER BY a.Provincia, a.Departamento;
"""

quedamos_con_max = db.query(QUEDAMOS_CON_MAX).df()

#%%

# Por último ponemos el nombre correcto de Clae
SQL4 = """
    SELECT Provincia, Departamento,
        CASE WHEN LENGTH(CAST(Clae6 AS VARCHAR)) = 5 
            THEN SUBSTR('0' || CAST(Clae6 AS VARCHAR), 1, 3)
            ELSE SUBSTR(CAST(Clae6 AS VARCHAR), 1, 3)
        END AS Clae3,
        "Cant. empleos"
    FROM quedamos_con_max
    ORDER BY Provincia, Departamento;
"""
sql4 = db.query(SQL4).df()


#%%

#Lo descargamos para el informe
#sql4.to_csv('~/LabodeDatos/TP01/ConsultaSQL4.csv', index = False)

