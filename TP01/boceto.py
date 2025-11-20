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

#%%Código para ubicar las DB

direccion_actual = pl.Path(__file__).parent.resolve()
str_dir = str(direccion_actual) 



#%%                             Lectura de datos
#%% EDUCATIVO
#%%

EE_df = pd.read_excel(str_dir+'/TablasOriginales/2022_padron_oficial_establecimientos_educativos.xlsx', 
                         skiprows=6, na_values=' ')
#skiprows saltea las primeras 6 filas, tienen información irrelevante
#na_values setea que todos los valores str == ' ' a nan

#%% PRODUCTIVO
#%%

EP_df = pd.read_csv(str_dir+'/TablasOriginales/Datos_por_departamento_actividad_y_sexo.csv')

#%% POBLACION
#%%
poblacion = pd.read_excel(str_dir+"/TablasOriginales/padron_poblacion.xlsX")



#%%                                 GQM 
#%% EP anio
#%%
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
EP_df = db.query(solucion_anio).df()


#%% EE Telefono
#%%

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
#%%

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
#%%

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


#%% Provincias entre EE y EP 
#%%

# Primero vemos en que difieren los valores de Jurisdicción de EE_df con 
#respecto a provincia de EP_df. Luego al revés.

Consulta1_dif_jurisdicción_provincia = """
    SELECT DISTINCT ee.Jurisdicción, ep.provincia
    FROM EE_df AS ee
    LEFT OUTER JOIN EP_df AS ep
    ON ep.provincia = ee.Jurisdicción
    GROUP BY ee.Jurisdicción, ep.provincia
"""

consulta1 = db.query(Consulta1_dif_jurisdicción_provincia).df()

Consulta2_dif_jurisdicción_provincia = """
    SELECT DISTINCT ee.Jurisdicción, ep.provincia
    FROM EE_df AS ee
    RIGHT OUTER JOIN EP_df AS ep
    ON ep.provincia = ee.Jurisdicción
    GROUP BY ee.Jurisdicción, ep.provincia
"""

consulta2 = db.query(Consulta2_dif_jurisdicción_provincia).df()

# Observamos que los nombres de las provincias difieren por tildes y mayúsculas,
#y en el caso de CABA por Ciudad Autónoma de Buenos Aires

#%%

# Conseguimos la métrica

Metrica1 =  """
        SELECT COUNT(*) AS Cantidad_Nulls
        FROM consulta2
        WHERE Jurisdicción IS NULL
        GROUP BY Jurisdicción
        """
metrica1 = db.query(Metrica1).df()


m1 = metrica1.loc[0,"Cantidad_Nulls"] / 24
print(m1)

#%%

#Entonces cambiamos las letras de las provincias a mayúsculas

Mayuscula_provincia_EP = """
    SELECT DISTINCT anio, in_departamentos, 
    departamento, 
    provincia_id, 
    UPPER(provincia) AS provincia, 
    clae6, 
    clae2, 
    letra, 
    genero, 
    Empleo, 
    Establecimientos, 
    empresas_exportadoras
    FROM EP_df
"""

EP_df = db.query(Mayuscula_provincia_EP).df()

Mayuscula_provincia_EE = """
    SELECT DISTINCT UPPER(Jurisdicción) AS Jurisdicción, Cueanexo, Nombre, Sector, Ámbito, Domicilio,
           "C. P.", "Código de área", Teléfono, "Código de localidad",
           Localidad, Departamento, Mail, Común, Especial, Adultos,
           Artística, Hospitalaria, Intercultural, Encierro,
           "Nivel inicial - Jardín maternal", "Nivel inicial - Jardín de infantes",
           Primario, Secundario, "Secundario - INET", SNU, "SNU - INET",
           "Secundario.1", "SNU.1", Talleres,
           "Nivel inicial - Educación temprana",
           "Nivel inicial - Jardín de infantes.1", "Primario.1", "Secundario.2",
           "Integración a la modalidad común/ adultos", "Primario.2",
           "Secundario.3", Alfabetización, "Formación Profesional",
           "Formación Profesional - INET", Inicial, "Primario.3", "Secundario.4",
           "Unnamed: 43"
    FROM EE_df
"""

EE_df = db.query(Mayuscula_provincia_EE).df()

#%%

# Cambiamos el nombre de Ciudad de Buenos Aires a CABA en EE

Caba_Jurisdicciones = """
    SELECT DISTINCT REPLACE(Jurisdicción, 'CIUDAD DE BUENOS AIRES', 'CABA') AS Jurisdicción, Cueanexo, Nombre, Sector, Ámbito, Domicilio,
           "C. P.", "Código de área", Teléfono, "Código de localidad",
           Localidad, Departamento, Mail, Común, Especial, Adultos,
           Artística, Hospitalaria, Intercultural, Encierro,
           "Nivel inicial - Jardín maternal", "Nivel inicial - Jardín de infantes",
           Primario, Secundario, "Secundario - INET", SNU, "SNU - INET",
           "Secundario.1", "SNU.1", Talleres,
           "Nivel inicial - Educación temprana",
           "Nivel inicial - Jardín de infantes.1", "Primario.1", "Secundario.2",
           "Integración a la modalidad común/ adultos", "Primario.2",
           "Secundario.3", Alfabetización, "Formación Profesional",
           "Formación Profesional - INET", Inicial, "Primario.3", "Secundario.4",
           "Unnamed: 43"
           FROM EE_df
"""

EE_df = db.query(Caba_Jurisdicciones).df()

#%%

# Limpio lo que son acentos de las Jurisdicciones (provincias) EE
Acentos_jurisdiccion_EE =  """
            SELECT 
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(Jurisdicción, 'Á', 'A'),'É', 'E'),'Í', 'I'),
            'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS Jurisdicción, Cueanexo, Nombre, Sector, Ámbito, Domicilio,
                   "C. P.", "Código de área", Teléfono, "Código de localidad",
                   Localidad, Departamento, Mail, Común, Especial, Adultos,
                   Artística, Hospitalaria, Intercultural, Encierro,
                   "Nivel inicial - Jardín maternal", "Nivel inicial - Jardín de infantes",
                   Primario, Secundario, "Secundario - INET", SNU, "SNU - INET",
                   "Secundario.1", "SNU.1", Talleres,
                   "Nivel inicial - Educación temprana",
                   "Nivel inicial - Jardín de infantes.1", "Primario.1", "Secundario.2",
                   "Integración a la modalidad común/ adultos", "Primario.2",
                   "Secundario.3", Alfabetización, "Formación Profesional",
                   "Formación Profesional - INET", Inicial, "Primario.3", "Secundario.4",
                   "Unnamed: 43"
            FROM EE_df
            """
EE_df = db.query(Acentos_jurisdiccion_EE).df()

#%%

# Limpiamos lo que son acentos de provincia de EP
Acentos_provincia_EP =  """
            SELECT 
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(provincia, 'Á', 'A'),'É', 'E'),'Í', 'I'),
            'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS provincia, anio, in_departamentos, 
            departamento, 
            provincia_id, 
            clae6, 
            clae2, 
            letra, 
            genero, 
            Empleo, 
            Establecimientos, 
            empresas_exportadoras
            FROM EP_df
            
            """
EP_df = db.query(Acentos_provincia_EP).df()



#%%                             Limpieza de datos
#%% Limpieza establecimientos_educativos
#%% 

#Construyo un DF con las columnas de EE que nos sirven para nuestro problema
EE_limpio = EE_df[['Cueanexo', 'Jurisdicción','Departamento','Común', 'Nivel inicial - Jardín de infantes',
                                   'Primario', 'Secundario']]
#Elimino columnas repetidas: https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
EE_limpio = EE_limpio.loc[:,~EE_limpio.columns.duplicated()].copy()
    
#%% 

# Reemplazamos null
EE_limpio['Común'].replace(' ', np.nan, inplace = True)

#%% 

# Limpieza de filas que no pertenezcan a la modalidad común
EE_limpio = EE_limpio.dropna(subset = ['Común'])

#%% 

# Descarto la columna 'Común' porque no aporta información.
EE_limpio = EE_limpio.drop(['Común'], axis = 1)

#%%

# Renombro las columnas 'Nivel inicial - Jardín de infantes', 
EE_limpio = EE_limpio.rename(columns = {'Nivel inicial - Jardín de infantes': 'Jardin'})

#%% 

# Cambiamos los valores null de las siguientes tres columnas para que el dominio
#de esos tres atributos sean booleanos [0,1]
EE_limpio = EE_limpio[['Cueanexo', 'Jurisdicción', 'Departamento', 'Jardin', 'Primario', 'Secundario']].fillna(0)

#%% 

# Convierto los nombres de los departamentos a mayuscula 
mayus = """
        SELECT Jurisdicción as Provincia, UPPER(Departamento) AS Departamento, Cueanexo, Jardin,
                                   Primario, Secundario
        FROM EE_limpio
        """
EE_limpio =  db.query(mayus).df()

#%% 

# Modificamos los nombres de Departamento para que coincidan con EP
#limpio lo que son acentos
acentos_departamento =  """
            SELECT Provincia,
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(Departamento, 'Á', 'A'),'É', 'E'),'Í', 'I'),
            'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS Departamento, Cueanexo, Jardin, Primario, Secundario
            FROM EE_limpio
            
            """
EE_limpio = db.query(acentos_departamento).df()

#%% 

# Tomo como referencias los nombre de departamentos y provincias de E. Productivos y edito los de E.Educativos  ----> (lo que tengo, lo que quiero)      
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("1§ DE MAYO", "1° DE MAYO")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("GENERAL ANGEL V PEÑALOZA", "ANGEL VICENTE PEÑALOZA")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("CORONEL DE MARINA L ROSALES", "CORONEL DE MARINA LEONARDO ROSALES")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("CORONEL FELIPE VARELA", "GENERAL FELIPE VARELA")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("DOCTOR MANUEL BELGRANO", "DR. MANUEL BELGRANO")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("GENERAL JUAN F QUIROGA", "GENERAL JUAN FACUNDO QUIROGA")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("GENERAL JUAN MARTIN DE PUEYRREDON", "JUAN MARTIN DE PUEYRREDON")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("GENERAL OCAMPO",  "GENERAL ORTIZ DE OCAMPO")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("JUAN B ALBERDI", "JUAN BAUTISTA ALBERDI")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("JUAN F IBARRA", "JUAN FELIPE IBARRA"  )
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("MAYOR LUIS J FONTANA", "MAYOR LUIS J. FONTANA")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("O HIGGINS", "O'HIGGINS")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("LIBERTADOR GRL SAN MARTIN" , "LIBERTADOR GENERAL SAN MARTIN")
EE_limpio["Departamento"] = EE_limpio["Departamento"].replace("O HIGGINS", "O'HIGGINS")


#%% Limpieza Establecimientos_Productivos
#%% 

# Nos quedamos con las columnas que nos sirven para nuestro sistema
EP_limpio = EP_df[['departamento', 'in_departamentos', 'provincia', 'provincia_id', 'clae6', 'genero', 
                       'Empleo', 'Establecimientos', 'empresas_exportadoras']]

#%% 

# Asigno los nombres de los atributos como en el Modelo Relacional, y 
#modificamos los valores de departamento para que esten en mayusculas
consulta_EP = """
        SELECT provincia AS Provincia, 
        provincia_id,
        UPPER(departamento) AS Departamento,
        in_departamentos,
        clae6 AS Clae6,
        genero as Sexo,
        Empleo AS Empleados,
        Establecimientos,
        empresas_exportadoras AS Empresas_exportadoras
        FROM EP_limpio
"""

EP_limpio = db.query(consulta_EP).df()

#%% 

# Modificamos los nombres de Departamento para que no tengan tildes
acentos_departamento_EP =  """
            SELECT Provincia, provincia_id,
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(
            REPLACE(Departamento, 'Á', 'A'),'É', 'E'),'Í', 'I'),
            'Ó', 'O'),'Ú', 'U'),'Ü', 'U') AS Departamento, in_departamentos, Clae6, Sexo, Empleados, Establecimientos, Empresas_exportadoras
            FROM EP_limpio
            
            """
EP_limpio = db.query(acentos_departamento_EP).df()


#%% Limpieza padrón_población 
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
        codigos.append(int(codigo)) # me guardo el id de la provincia que es el primero o los primeros dos digitos


#%% 

# Teniendo los indices recorro por tablas y me guardo los datos 
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
    id_dep = codigos[i]
    
    
    
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
         {"id_Departamento": id_dep,
          "Departamento": depto,
          "Población_jardin": jardin,
          "Población_primario": primaria,
          "Población_secundario": secundaria,
          "Cantidad_habitantes": total}
        )

#al finalizar el for obtengo una lista de diccionarios, solo queda convertirlo a DataFrame
poblacion_limpio = pd.DataFrame(resultados)



#%%                              Modelado de DB
#%% Provincia
#%%

# Creamos Provincia
CrearProvincia = """
    SELECT DISTINCT provincia_id AS id_Provincia, Provincia AS Nombre_provincia
    FROM EP_limpio
    GROUP BY provincia_id, Provincia
"""

Provincia = db.query(CrearProvincia).df()


#%% Departamento
#%% 

# Las siguientes son consultas para investigar sobre los departamentos en cada
#fuente de datos

#Apartamos departamentos y provincias de EE
departamentos_de_EE = """
        SELECT DISTINCT Provincia, Departamento
        FROM EE_limpio
        GROUP BY Provincia,Departamento
        ORDER BY Provincia ASC
      """
departamentos_de_EE = db.query(departamentos_de_EE).df()

#%%

# Apartamos departamentos y provincias con su respectivo ID de EP
departamentos_de_EP_y_id = """
        SELECT DISTINCT in_departamentos, UPPER(departamento) AS Departamento,
        Provincia, provincia_id
        FROM EP_limpio
        GROUP BY in_departamentos, departamento, provincia, provincia_id
        ORDER BY provincia ASC
      """
departamentos_de_EP_y_id = db.query(departamentos_de_EP_y_id).df()

#%%

# Buscamos diferencias entre EE y EP
diferencias_departamentos_EE_a_EP = """
        SELECT DISTINCT a.Provincia, a.Departamento, 
        b.Provincia, UPPER(b.Departamento)      
        FROM departamentos_de_EE AS a
        LEFT OUTER JOIN departamentos_de_EP_y_id AS b
        ON a.Provincia = b.provincia AND a.Departamento = b.Departamento
        WHERE b.Provincia IS NULL
        """
diferencias_departamentos_EE_a_EP = db.query(diferencias_departamentos_EE_a_EP).df() 

diferencias_departamentos_EP_a_EE = """
        SELECT DISTINCT a.Provincia, a.Departamento, 
        b.Provincia, UPPER(b.Departamento) AS Departamento   
        FROM departamentos_de_EE AS a
        RIGHT OUTER JOIN departamentos_de_EP_y_id AS b
        ON a.Provincia = b.Provincia AND a.Departamento = b.Departamento
        WHERE a.Provincia IS NULL
        """
diferencias_departamentos_EP_a_EE = db.query(diferencias_departamentos_EP_a_EE).df() 

# Podemos ver que dentro de Tierra del Fuego, el departamento Tolhuin no 
#posee ningún establecimiento educativo, y el departamento Antartida Argentina
#no posee ningún establecimiento productivo.

#%%

# Concluimos que hay 528 Departamentos diferentes  
# Creamos Departamento
CrearDepartamento = """
                    SELECT DISTINCT in_departamentos AS id_Departamento, Departamento AS Nombre_depto,
                        provincia_id AS id_Provincia
                    FROM EP_limpio
                    GROUP BY id_Departamento, Nombre_depto, id_Provincia
                    """
Departamento = db.query(CrearDepartamento).df()

#%%

# Como creamos Departamento a partir de los Ids de los Departamentos de los 
#Establecimientos PRoductivos, entonces no tenemos la información del 
#Departamento Antartida Argentina que es el único Departamento que no tiene
#Establecimientos Productivos. Por ese motivo lo agregamos a la relación
#Departamento y le colocamos el id 1    
Departamento.loc[len(Departamento)] = [1, "ANTARTIDA ARGENTINA", 94]


#%% Establecimientos_Educativos
#%%

#Creamos Establecimientos_Educativos
Creamos_Establecimientos_Educativos = """
    SELECT DISTINCT ep.in_departamentos AS id_Departamento, ee.Cueanexo, ee.Jardin ,
    ee.Primario AS Primaria, ee.Secundario AS Secundaria
    FROM EE_limpio AS ee
    LEFT OUTER JOIN EP_limpio AS ep
    ON ep.Departamento = ee.Departamento AND ep.Provincia = ee.Provincia
    ORDER BY Cueanexo ASC
"""

Establecimientos_Educativos = db.query(Creamos_Establecimientos_Educativos).df()

# Insertamos el id_Departamento código numero 1 al único colegio de Antartida
#Argentina de cueanexo 940011700. 
Establecimientos_Educativos.iloc[50221,0] = 1
 

#%% Establecimientos_Productivos
#%%

# Creamos Establecimientos_Productivos
Crear_Establecimientos_Productivos = """
    SELECT DISTINCT in_departamentos AS id_Departamento, Clae6, Sexo, 
    Empleados, Establecimientos, Empresas_exportadoras
    FROM EP_limpio
"""

Establecimientos_Productivos =  db.query(Crear_Establecimientos_Productivos).df()
 

#%% Población
#%% 

# Seleccionamos el departamento y su ID en poblacion_limpio para comparar entre ids de
#departamentos de EP
departamentos_de_poblacion_y_ID = """
            SELECT DISTINCT id_Departamento, Departamento
            FROM poblacion_limpio
            GROUP BY id_Departamento, Departamento
"""

departamentos_de_poblacion_y_ID = db.query(departamentos_de_poblacion_y_ID).df()

#%%

# Buscamos diferencias de departamentos entre id de poblacion e id de EP
diferencias_departamentos_poblacion_a_EP = """
        SELECT DISTINCT a.id_departamento, UPPER(a.Departamento) AS Departamento,
        b.in_departamentos, UPPER(b.Departamento) AS Departamento
        FROM departamentos_de_poblacion_y_ID AS a
        LEFT OUTER JOIN departamentos_de_EP_y_id AS b
        ON in_departamentos = id_departamento
        WHERE in_departamentos IS NULL
        """
diferencias_departamentos_poblacion_a_EP = db.query(diferencias_departamentos_poblacion_a_EP).df()

diferencias_departamentos_EP_a_poblacion = """
        SELECT DISTINCT a.id_departamento, UPPER(a.Departamento) AS Departamento,
        b.in_departamentos, UPPER(b.Departamento) AS Departamento
        FROM departamentos_de_poblacion_y_ID AS a
        RIGHT OUTER JOIN departamentos_de_EP_y_id AS b
        ON in_departamentos = id_departamento
        WHERE id_Departamento IS NULL
        """
diferencias_departamentos_EP_a_poblacion = db.query(diferencias_departamentos_EP_a_poblacion).df()

#%%  

# primero modificamos manuelamente los codigos que decidimos tomarlos como 
# que hacen referencia a los mismos departamentos
poblacion_limpio.iloc[526,0] = 94014
poblacion_limpio.iloc[524,0] = 94007
poblacion_limpio.iloc[45,0] = 6217

# Creamos poblacion
Población = """
            SELECT DISTINCT id_Departamento, Población_jardin, 
            Población_primario, Población_secundario, Cantidad_habitantes
            FROM poblacion_limpio
            """
Población = db.query(Población).df()


#%%                           Consultas SQL
#%% 1) SQL
#%%

# Buscamos los ids de Departamentos con establecimientos educativos, y contamos
#cuantos establecimientos tienen de cada modalidad
id_deptos_con_EE_por_modalidad = """
    SELECT DISTINCT id_Departamento,
    SUM(CASE WHEN Jardin = 1 THEN 1 ELSE 0 END) Jardines,
    SUM(CASE WHEN Primaria = 1 THEN 1 ELSE 0 END) AS Primarias,
    SUM(CASE WHEN Secundaria = 1 THEN 1 ELSE 0 END) AS Secundarios,
FROM Establecimientos_Educativos
GROUP BY id_Departamento
ORDER BY id_Departamento
"""

id_deptos_con_ee_por_modalidad = db.query(id_deptos_con_EE_por_modalidad).df()

#%%

# Obtenemos csv
DEPARTAMENTO_PROVINCIA_ID_DEPARTAMENTO =  """
    SELECT p.Nombre_provincia AS Provincia, d.Nombre_depto AS Departamento,
    d.id_Departamento
    FROM Provincia AS p
    INNER JOIN Departamento AS d
    ON p.id_Provincia = d.id_Provincia
"""

departamento_provincia_id_departamento = db.query(DEPARTAMENTO_PROVINCIA_ID_DEPARTAMENTO).df()

#%%

# Nos quedamos con todos los departamentos y rellenamos con null los valores que
#no tengan ee
UNION_SQL1 = """
    SELECT d.id_Departamento, d.Provincia, d.Departamento, i.Jardines,
    i.Primarias, i.Secundarios
    FROM departamento_provincia_id_departamento AS d
    LEFT OUTER JOIN id_deptos_con_ee_por_modalidad AS i
    ON i.id_Departamento = d.id_Departamento
"""
union_sql1 = db.query(UNION_SQL1).df()

#%%

# Ahora nos quedamos con todos los departamentos y rellenamos con null los 
#valores que no tengan poblacion
UNION_SQL2 = """
    SELECT DISTINCT u.Provincia, u.Departamento,
    u.Jardines,
    p.Población_jardin AS "Población Jardin",
    u.Primarias,
    p.Población_primario AS "Población Primaria",
    u.Secundarios,
    p.Población_secundario AS "Población Secundaria"
    FROM union_sql1 AS u
    LEFT OUTER JOIN Población AS p
    ON p.id_Departamento = u.id_Departamento
"""

union_sql1 = db.query(UNION_SQL2).df()

#%%

# Por último rellenamos con 0 los valores null
SQL1 = """
    SELECT DISTINCT Provincia, Departamento,
    CASE WHEN Jardines IS NULL THEN 0 ELSE Jardines END AS Jardines,
    CASE WHEN "Población Jardin" IS NULL THEN 0 ELSE "Población Jardin" END AS "Población Jardin",
    CASE WHEN Primarias IS NULL THEN 0 ELSE Primarias END AS Primarias,
    CASE WHEN "Población Primaria" IS NULL THEN 0 ELSE "Población Primaria" END AS "Población Primaria",
    CASE WHEN Secundarios IS NULL THEN 0 ELSE Secundarios END AS Secundarios,
    CASE WHEN "Población Secundaria" IS NULL THEN 0 ELSE "Población Secundaria" END AS "Población Secundaria"
    FROM union_sql1
    ORDER BY Provincia ASC, Primarias DESC
"""

sql1 = db.query(SQL1).df()


#%% 2) SQL
#%%

# Conseguimos la cantidad total de empleados por departamento
EMPLEADOS_POR_DEPARTAMENTO = """
    SELECT id_Departamento, 
    SUM(Empleados) AS "Cantidad total de empleados en 2022"
    FROM Establecimientos_Productivos
    GROUP BY id_Departamento
"""

empleados_por_departamento = db.query(EMPLEADOS_POR_DEPARTAMENTO).df()

#%%

# Hacemos un Left JOIN con preponderancia en departamento_provincia_id_departamento
UNION_SQL2 = """
    SELECT d.Provincia, d.Departamento, e."Cantidad total de empleados en 2022"
    FROM departamento_provincia_id_departamento AS d
    LEFT OUTER JOIN empleados_por_departamento AS e
    ON d.id_Departamento = e.id_Departamento
"""

union_sql2 = db.query(UNION_SQL2).df()

#%%

# Terminamos sql2
SQL2 = """
    SELECT Provincia, Departamento, 
    CASE WHEN "Cantidad total de empleados en 2022" IS NULL THEN 0 ELSE "Cantidad total de empleados en 2022" END AS "Cantidad total de empleados en 2022"
    FROM union_sql2
    ORDER BY Provincia ASC, "Cantidad total de empleados en 2022" DESC
"""

sql2 = db.query(SQL2).df()


#%% 3) SQL
#%%

# Contamos la cantidad de empresas exportadoras que emplean mujeres
EMPRESAS_EXP_MUJERES = """
    SELECT DISTINCT id_Departamento, 
    SUM(CASE WHEN Sexo = 'Mujeres' THEN Empresas_exportadoras ELSE 0 END) AS Cant_Expo_Mujeres, 
    FROM Establecimientos_Productivos
    GROUP BY id_Departamento
"""

empresas_exp_mujeres = db.query(EMPRESAS_EXP_MUJERES).df()
#%%

# Contamos cantidad de Establecimientos Educativos por Departamento
CANTIDAD_EE = """
    SELECT id_Departamento, COUNT(*) AS Cant_EE
    FROM Establecimientos_Educativos
    GROUP BY id_Departamento
"""

cantidad_ee = db.query(CANTIDAD_EE).df()

#%%%

# Vamos ensamblando valores a la variable union_sql3 que terminará siendo la
#respuesta a la consigna
UNION_SQL3 = """
    SELECT d.id_Departamento, d.Provincia, d.Departamento, 
    p.Cantidad_habitantes AS Población
    FROM departamento_provincia_id_departamento AS d
    LEFT OUTER JOIN Población AS p
    ON d.id_Departamento = p.id_Departamento
"""

union_sql3 = db.query(UNION_SQL3).df()

#%%

# Ahora agregamos los valores de empresas_exp_mujeres
UNION_SQL3_2 = """
    SELECT u.id_Departamento, u.Provincia, u.Departamento,
    e.Cant_Expo_Mujeres , u.Población
    FROM union_sql3 AS u
    LEFT OUTER JOIN empresas_exp_mujeres AS e
    ON u.id_Departamento = e.id_Departamento
"""

union_sql3 = db.query(UNION_SQL3_2).df()

#%%

# Por último ensamblamos los valores de cantidad_ee
UNION_SQL3_3 = """
    SELECT u.Provincia, u.Departamento, u.Cant_Expo_Mujeres, e.Cant_EE,
    u.Población
    FROM union_sql3 AS u
    LEFT OUTER JOIN cantidad_ee AS e
    ON u.id_Departamento = e.id_Departamento
"""

union_sql3 = db.query(UNION_SQL3_3).df()

#%%

# Finalmente cambiamos valores null por 0
SQL3 = """
    SELECT DISTINCT Provincia, Departamento,
    CASE WHEN Cant_Expo_Mujeres IS NULL THEN 0 ELSE Cant_Expo_Mujeres END AS Cant_Expo_Mujeres,
    CASE WHEN Cant_EE IS NULL THEN 0 ELSE Cant_EE END AS Cant_EE,
    CASE WHEN Población IS NULL THEN 0 ELSE Población END AS Población
    FROM union_sql3
    ORDER BY Cant_EE DESC, Cant_Expo_Mujeres DESC, Provincia ASC, Departamento ASC;
"""

sql3 = db.query(SQL3).df()


#%% 4) SQL
#%%

# Unimos empleados_x_departamento con su provincia
UNION_PROVINCIA_EMPLEADOS = """
    SELECT e.id_Departamento, d.Provincia, d.Departamento, 
    e."Cantidad total de empleados en 2022" AS "Cant. empleos"
    FROM empleados_por_departamento AS e
    INNER JOIN departamento_provincia_id_departamento AS d
    ON e.id_Departamento = d.id_Departamento 
"""

union_provincia_empleados = db.query(UNION_PROVINCIA_EMPLEADOS).df()

#%%

# Seleccionamos los departamentos que por cada provincia tienen mayor 
#cantidad de empleos promedio de su provincia en base a 
#union_provincia_empleados
PROMEDIO = """
    SELECT id_Departamento, Provincia, Departamento, "Cant. empleos"
    FROM union_provincia_empleados AS c1
    WHERE "Cant. empleos" > (
        SELECT AVG("Cant. empleos")
        FROM union_provincia_empleados AS c2
        WHERE c2.Provincia = c1.Provincia)
    ORDER BY Provincia, "Cant. empleos" DESC;
"""

promedio = db.query(PROMEDIO).df()

#%%

# Uno Establecimientos_Productivos con promedio para que me queden los
#departamentos con cantidad de empleados mayor al promedio
LIMPIEZA_EP_MAYOR_PROMEDIO = """
    SELECT p.id_Departamento, p.Provincia, p.Departamento, ep.Clae6, 
    ep.Sexo, ep.Empleados
    FROM Establecimientos_Productivos AS ep
    INNER JOIN promedio as p
    ON ep.id_Departamento = p.id_Departamento;
"""

limpieza_EP_mayor_promedio = db.query(LIMPIEZA_EP_MAYOR_PROMEDIO).df()

#%%

# Agrupamos hombres y mujeres en "Cant. empleos" por clae por departamento
AGRUPAMOS_X_SEXO = """
    SELECT id_Departamento, Provincia, Departamento, Clae6, SUM(Empleados) AS "Cant. empleos"
    FROM limpieza_EP_mayor_promedio
    GROUP BY id_Departamento, Provincia, Departamento, Clae6
"""

agrupamos_x_sexo = db.query(AGRUPAMOS_X_SEXO).df()

#%%

# Nos quedamos con el clae6 que mas empleo genera en los departamentos
#seleccionados anteriormente
QUEDAMOS_CON_MAX = """
    SELECT a.id_Departamento, a.Provincia, a.Departamento, a.Clae6, a."Cant. empleos"
    FROM agrupamos_x_sexo AS a
    INNER JOIN (
        SELECT id_Departamento, Provincia, Departamento, 
        MAX("Cant. empleos") AS max_empleos
        FROM agrupamos_x_sexo
        GROUP BY id_Departamento, Provincia, Departamento) AS b
    ON a.id_Departamento = b.id_Departamento AND a."Cant. empleos" = b.max_empleos
"""

quedamos_con_max = db.query(QUEDAMOS_CON_MAX).df()

#%%

# Por último terminamos con nuestro ejercicio con la siguiente función. Como a
#partir de la clase no lograbamos entender como remplazar los códigos de clae3
#para claes6 con 5 dígitos, acudimosa la librería de DuckDB:
# https://duckdb.org/docs/stable/sql/functions/text?utm_source=chatgpt.com
SQL4 = """
    SELECT Provincia, Departamento,
    CASE WHEN LENGTH(CAST(Clae6 AS VARCHAR)) = 5 
    THEN SUBSTR('0' || CAST(Clae6 AS VARCHAR), 1, 3)
    ELSE SUBSTR(CAST(Clae6 AS VARCHAR), 1, 3) END AS CLAE3,
    "Cant. empleos"
    FROM quedamos_con_max
    ORDER BY Provincia, Departamento
"""

sql4 = db.query(SQL4).df()



#%%                                 Gráficos
#%% Gráfico 1)
#%%

C5 =    """
        SELECT Provincia, SUM (Empleados) AS "Empleados por provincia"
        FROM Establecimientos_Productivos
        GROUP BY Provincia
        ORDER BY "Empleados por provincia" DESC
        """
            
Grafico1 = db.query(C5).df()

Grafico1.plot(
    x = "Provincia",
    y = "Empleados por provincia",
    kind = 'bar',
    #yticks = 10000,
    xlabel = 'Provincia',
    title = 'Cantidad de empleados por provincia en 2022',
    )

#%% Gráfico 2)


#sql1
res = []
for i in range(len(sql1)):
    Provincia = sql1.iloc[i,0]
    Departamento = sql1.iloc[i,1]
    Jardines = sql1.iloc[i,2]
    Primarios = sql1.iloc[i,4]
    Secundarios = sql1.iloc[i,6]
    PoblacionJ = sql1.iloc[i,3]
    PoblacionP = sql1.iloc[i,5]
    PoblacionS = sql1.iloc[i,7]
    
    j = {"Provincia":Provincia,"Departamento":Departamento,"Nivel":"Jardin","cant_EE":Jardines,"Poblacion":PoblacionJ}
    p = {"Provincia":Provincia,"Departamento":Departamento,"Nivel":"Primario","cant_EE":Primarios,"Poblacion":PoblacionP}
    s = {"Provincia":Provincia,"Departamento":Departamento,"Nivel":"Secundario","cant_EE":Secundarios,"Poblacion":PoblacionS}
    
    res.append(j)
    res.append(p)
    res.append(s)
    
    
dato_graf = pd.DataFrame(res)

    
sns.scatterplot(
    data = dato_graf,
    x = "Poblacion",
    y = "cant_EE",
    hue = "Nivel")


#%% Gráfico 3)
#%%

# Hacemos una consulta para obtener en un DataFrame llamado 
#df_figura4 en el cual obtenemos las provincias para 
#cantidad_ee
DF_FIGURA3 = """
    SELECT e.id_Departamento, d.Provincia, d.Departamento, e.Cant_EE
    FROM cantidad_ee AS e
    INNER JOIN departamento_provincia_id_departamento AS d
    ON e.id_Departamento = d.id_Departamento
"""

df_figura3 = db.query(DF_FIGURA3).df()

#%%

# Obtenemos los códigos de las provincias
DF_FIGURA3_1 = """
    SELECT DISTINCT p.id_Provincia, d.Provincia, d.Departamento, d.cant_EE
    FROM df_figura3 AS d
    INNER JOIN Provincia AS p
    ON d.Provincia = p.Nombre_provincia
"""

df_figura3 = db.query(DF_FIGURA3_1).df()

#%%

# Hacemos un calculo de la mediana para cada provincia, para
#luego insertarlo en el hiperparámetro Order
MEDIANA = """
    SELECT id_Provincia, Provincia, MEDIAN(Cant_EE) AS Mediana_EE
    FROM df_figura3
    GROUP BY id_Provincia, Provincia
    ORDER BY Mediana_EE
"""

mediana = db.query(MEDIANA).df()

orden = mediana["id_Provincia"].values

#%%


fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(
    data = df_figura3,
    x = "id_Provincia",
    y = "Cant_EE",
    ax = ax,
    showfliers = False, #Eliminamos outliers para un mas legible gráfico
    order = orden # orden de medianas
)

ax.set_xlabel("Provincias", fontsize=11)
ax.set_ylabel("Cantidad de Establecimientos Educativos", fontsize=11)
ax.set_title("Boxplot de la cantidad de Establecimientos Educativos por Provincias", fontsize=13, fontweight="bold")
ax.grid(True)

plt.tight_layout() #ajusta automáticamente los espacios y márgenes del gráfico para que los títulos
plt.show()


#%%








#%% AUXILIAR

# Conseguimos los departamentos que no estan en Población
DEPARTAMENTOS_NO_EN_POBLACION = """
    SELECT DISTINCT d.id_Departamento
    FROM Departamento AS d
    LEFT OUTER JOIN Población AS p
    ON d.id_Departamento = p.id_Departamento
    WHERE p.id_Departamento IS NULL
"""

departamentos_no_en_poblacion = db.query(DEPARTAMENTOS_NO_EN_POBLACION).df()

# Ahora conseguimos los departamentos que no estan en Establecimientos_Educativos
DEPARTAMENTOS_NO_EN_EE = """
    SELECT DISTINCT d.id_Departamento
    FROM Departamento AS d
    LEFT OUTER JOIN Establecimientos_Educativos AS ee
    ON d.id_Departamento = ee.id_Departamento
    WHERE ee.id_Departamento IS NULL
"""
departamentos_no_en_ee = db.query(DEPARTAMENTOS_NO_EN_EE).df()




