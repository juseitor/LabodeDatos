import pandas as pd
import duckdb as dd

carpeta = "~/LabodeDatos/clase6/archivosclase6/"

#%%



print(carpeta)

empleado = pd.read_csv(carpeta+"empleado.csv")

consultaSQL = """
            SELECT DISTINCT DNI, Salario
            FROM empleado;
            """
            
dataframeResultado = dd.query(consultaSQL).df()#ejecuta la consulta
print(dataframeResultado)

#%%

#SQL Ejercicio 1.1

aeropuerto = pd.read_csv(carpeta+"aeropuerto.csv")

c1 = """
    SELECT DISTINCT Codigo, Nombre
    FROM aeropuerto
    WHERE Ciudad='Londres';
    """
            
df1 = dd.query(c1).df()#ejecuta la consulta
print(df1)

#SUUUUUUU

#%%

#SQL Ejercicio 1.3
vuelo      = pd.read_csv(carpeta+"vuelo.csv")    

c2 = """
SELECT DISTINCT Numero
FROM vuelo
WHERE Origen = 'CDG' AND Destino = 'LHR';
"""

df2 = dd.query(c2).df()
print(df2)

#%%

#Ejercicio 2

"""
SELECT DISTINCT *
FROM alumnos_BD
UNION
SELECT DISTINCT *
FROM alumnos_TLeng;
"""

"""
SELECT DISTINCT *
FROM alumnos_BD
INTERSECT
SELECT DISTINCT *
FROM alumnos_TLeng;
"""

"""
SELECT DISTINCT *
FROM alumnos_BD
EXCEPT
SELECT DISTINCT *
FROM alumnos_TLeng;
"""

#%%

#Ejercicio 2.1

vuelo      = pd.read_csv(carpeta+"vuelo.csv")   
reserva    = pd.read_csv(carpeta+"reserva.csv")

c3 = """
SELECT DISTINCT Nrovuelo
FROM reserva
INTERSECT
SELECT DISTINCT Numero
FROM vuelo;
"""

df3 = dd.query(c3).df()
print(df3)

#%%

#Ejercicio 2.2

vuelo      = pd.read_csv(carpeta+"vuelo.csv")   
reserva    = pd.read_csv(carpeta+"reserva.csv")

c4 = """
SELECT DISTINCT Numero
FROM vuelo
EXCEPT
SELECT DISTINCT Nrovuelo
FROM reserva;
"""

df4 = dd.query(c4).df()
print(df4)

#%%

#Ejercicio 2.3

vuelo      = pd.read_csv(carpeta+"vuelo.csv") 
aeropuerto = pd.read_csv(carpeta+"aeropuerto.csv")

c5 = """

"""


#%%

""" TODO
SELECT DISTINCT *
FROM PERSONA
CROSS JOIN NACIONALIDADES;
"""

""" JUNTA LAS QUE SON IGUAL SIMPLEMENTE
SELECT DISTINCT *
FROM PERSONA
INNER JOIN NACIONALIDADES
ON Nacionalidad=IDN;
"""

""" COMPLETA CON NULL
SELECT DISTINCT *
FROM PERSONA
LEFT OUTER JOIN NACIONALIDADES
ON Nacionalidad=IDN;
"""

#%%

# Ejercicio 3.1

vuelo      = pd.read_csv(carpeta+"vuelo.csv") 
aeropuerto = pd.read_csv(carpeta+"aeropuerto.csv")

c6 = """
SELECT DISTINCT Ciudad
FROM vuelo
INNER JOIN aeropuerto
ON Origen = Codigo
WHERE Numero = '165';
"""

df6 = dd.query(c6).df()
print(df6)

#%%

# Ejercicio 3.2

reserva    = pd.read_csv(carpeta+"reserva.csv")
pasajero   = pd.read_csv(carpeta+"pasajero.csv")

c7 = """
SELECT DISTINCT Nombre
FROM pasajero as p
INNER JOIN reserva as r
ON p.DNI = r.DNI
WHERE Precio < '200';
"""

df7 = dd.query(c7).df()
print(df7)

#%%

# Ejercicio 3.3

reserva    = pd.read_csv(carpeta+"reserva.csv")
pasajero   = pd.read_csv(carpeta+"pasajero.csv")
vuelo      = pd.read_csv(carpeta+"vuelo.csv") 
aeropuerto = pd.read_csv(carpeta+"aeropuerto.csv")

c8_1 = """
SELECT DISTINCT Numero, Origen, Destino
FROM vuelo
WHERE Origen = 'MAD'
"""
df8_1 = dd.query(c8_1).df()
print(df8_1)

c8_2 = """
SELECT DISTINCT DNI
FROM reserva
INNER JOIN df8_1
ON Numero = Nrovuelo;
"""

df8_2 = dd.query(c8_2).df()
print(df8_2)
#%%
# EJERCICIO 3.3 RESUELTO EN CLASE 

reserva    = pd.read_csv(carpeta+"reserva.csv")
pasajero   = pd.read_csv(carpeta+"pasajero.csv")
vuelo      = pd.read_csv(carpeta+"vuelo.csv") 
aeropuerto = pd.read_csv(carpeta+"aeropuerto.csv")

c = """ 
SELECT pasajero.DNI, Nombre, NroVuelo, Fecha
FROM pasajero
INNER JOIN reserva
ON pasajero.DNI = reserva.DNI
"""

df_1 = dd.query(c).df()

c = """
SELECT DNI, Nombre, Nrovuelo, Fecha, Origen, Destino
FROM df_1
INNER JOIN vuelo
ON df_1.Nrovuelo = vuelo.numero;
"""

df_2 = dd.query(c).df()
                                                                               
c = """
SELECT DNI, Nombre, Nrovuelo, Fecha, Origen, Destino, Ciudad as CiudadDestino
FROM df_2
INNER JOIN aeropuerto
ON df_2.destino = aeropuerto.codigo
"""

df_3 = dd.query(c).df()

c = """
SELECT DNI, Nombre, Nrovuelo, Fecha, Origen, Destino, CiudadDestino, Ciudad as CiudadOrigen
FROM df_3
INNER JOIN aeropuerto
ON df_3.Origen = aeropuerto.Codigo
"""

df_4 = dd.query(c).df()

c = """
SELECT Nombre, Fecha, CiudadDestino
FROM df_4
WHERE CiudadOrigen = 'Madrid'
"""
df_5 = dd.query(c).df()
#%%

# SQL Desafío 1

examen     = pd.read_csv(carpeta+"examen.csv")

a = """
SELECT Nombre, Sexo, Edad
FROM examen
GROUP BY Nombre, Sexo, Edad
"""

df1 = dd.query(a).df()

b = """
SELECT Nombre, Instancia, Nota
FROM examen
GROUP BY Nombre, Instancia, Nota
"""

df2 = dd.query(b).df()

c = """
SELECT Nombre, Sexo, Edad, Instancia, Nota
FROM df1
INNER JOIN df2
ON df1.Nombre = df2.Nombre
GROUP BY df1.Nombre
"""

df3 = dd.query(c).df()