#Tengo que escribir la ruta del archivo en cada celda
archivo = "/home/jusa/Escritorio/LabodeDatos/clase1-2/ejemplo.txt" #asi funciona bien

#%%
#manera automatica
#archivo = 'ejemplo.txt' #asi no funciono una vez que probe
archivo = "/home/jusa/Escritorio/LabodeDatos/clase1-2/ejemplo.txt" #asi funciona bien
with open(archivo, 'rt') as file:
    data =file.read()
    data
    print(data)
#%%
    
#%%
#Manera manual
archivo = "/home/jusa/Escritorio/LabodeDatos/clase1-2/ejemplo.txt" #asi funciona bien
with open(archivo,'rt') as file:
    data = file.read()
f = open(archivo, 'rt')
data = f.read()
f.close()
data
print(data)
#%%

#%%

#Escribir archivo
archivo = "/home/jusa/Escritorio/LabodeDatos/clase1-2/ejemplo.txt"
with open(archivo,'rt') as file:
    data = file.read()
    data_nuevo = 'inicio del texto' + data
    data_nuevo = data_nuevo + 'cierre del texto'
    
    datame = open("nuevonombre.txt",'w') #write mode
    datame.write (data_nuevo)
    datame.close()
#%%    