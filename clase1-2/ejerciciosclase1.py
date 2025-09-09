#from typing import List
import random

#Tengo que escribir la ruta del archivo en cada celda
archivo = "/home/jusa/Escritorio/LabodeDatos/clase1-2/ejemplo.txt" #asi funciona bien

#%% 1
#manera automatica
#archivo = 'ejemplo.txt' #asi no funciono una vez que probe
archivo = "/home/jusa/Escritorio/LabodeDatos/clase1-2/ejemplo.txt" #asi funciona bien
with open(archivo, 'rt') as file:
    data =file.read()
    data
    print(data)
#%% 2
#Manera manual
archivo = "/home/jusa/Escritorio/LabodeDatos/clase1-2/ejemplo.txt" #asi funciona bien
with open(archivo,'rt') as file:
    data = file.read()
f = open(archivo, 'rt')
data = f.read()
f.close()
data
print(data)
#%% 3
#Escribir archivo
archivo = "/home/jusa/Escritorio/LabodeDatos/clase1-2/ejemplo.txt"
with open(archivo,'rt') as file:
    data = file.read()
    data_nuevo = 'inicio del texto' + data
    data_nuevo = data_nuevo + 'cierre del texto'
    
    datame = open("nuevonombre.txt",'w') #write mode
    datame.write (data_nuevo)
    datame.close()
#%% 4

#import random

def generala_tirar1():
    # Genera una lista con 5 valores al azar entre 1 y 6
    tirada = [random.randint(1, 6) for _ in range(5)]
    return tirada

print(generala_tirar1())  

#%% 5

def generala_tirar2():
    tirada = []  # lista vacía donde guardaremos los dados
    
    for i in range(5):  # repetimos 5 veces
        dado = random.randint(1, 6)  # tiramos un dado
        tirada.append(dado)  # agregamos el valor a la lista
    return tirada

print(generala_tirar2())

#%% 6
#No devuelve nada. Desp tengo que verlo
def generala_completa() :
    tirada = [random.randint(1, 6) for _ in range(5)]
#    return tirada
    i : int = 0
    res : bool = True
    while res == True and i < 4 :
        if  tirada[i] < tirada[i+1]:
            res = True
            i = i + 1
        else:
            res = False
    if res == True:
        return tirada, "Escalera"
    else:
        return tirada, "No es escalera"
    print(generala_completa())
#%%
# Escribir un programa que recorra las líneas del archivo ‘datame.txt’ e imprima solamente las líneas que contienen la palabra ‘estudiante’

# Abrimos el archivo en modo lectura
texto = "/home/jusa/Escritorio/LabodeDatos/clase1-2/archivosclase1/datame.txt"
with open(texto, "r", encoding="utf-8") as archivo: #"utf-8" es una codificación estándar que puede representar prácticamente todos los caracteres (acentos, ñ, emojis, etc.).
    for linea in archivo:
        if "estudiante" in linea:   # buscamos la palabra
            print(linea.strip())    # .strip() quita saltos de línea extras
            
#%%

