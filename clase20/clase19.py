from bs4 import BeautifulSoup
import requests

url = "https://lcd.exactas.uba.ar/materias/"
r = requests.get(url)
print(r.text)

#%%

r = requests.get("https://lcd.exactas.uba.ar/materias/")
materias = BeautifulSoup(r.text, 'html.parser')
materias

#%%

materias.find('h1')

#%%

materias.find_all('h1')

#%% Para acceder a los valores de los atributos podemos usar el comando .get()

materias.find('h1').get('class')

#%%

print(materias.find('h1').get('style'))
print(materias.find('h2').get('style'))

#%% Podemos buscar elementos de la página buscando sus atributos como clase o estilo.

materias.find_all(attrs={'class':'flip-box-heading-back'})

#%%

materias.find_all('h3', attrs={'class':'flip-box-heading-back'})

#%%

materias.find_all('li', attrs={'style':True})

#%% A veces, los elementos pueden tener varias clases. Se pueden acceder a esos elementos consultando por cualquiera de las clases.

div = materias.find(attrs={'class':'fusion-layout-column'})
div.get('class')

#%% Podemos usar los comandos find() y find_all() en un elemento para buscar elementos que sean hijos del primero. 
#Intentemos acceder a todas las filas de la tabla del cronograma sugerido a partir de algún elemento padre.

div = materias.find('ul')   
div.find_all('li')

#%%

materias.find_all(attrs={'class':'fusion-column-wrapper fusion-content-layout-column'})

#%%

divs = materias.find_all(attrs={'class':'table'})
len(divs)

#%%

div = materias.find(attrs={'class':'table'})
rows = div.find_all('tr') #tr: table row
for row in rows:
  celdas = row.find_all('td')
  print(celdas)
  
#%%

div = materias.find(attrs={'class':'table'})
rows = div.find_all('tr') #tr: table row
for row in rows:
  celdas = row.find_all('td')
  print(celdas[0].text, celdas[1].text, celdas[2].text)
  
#%%



#%% EJEMPLO EN CLASE

rr = requests.get('https://datos.gob.ar/dataset?groups=agri')
agricultura = BeautifulSoup(rr.text, 'html.parser')
agricultura

#%% 

agricultura.find_all('h3')

#%% Notese casi todos los agricultura tienen todas class: "dataset-title"

agricultura.find('h3').get('class')

#%%

print(agricultura.find('h3').get('style'))

#%% El find_all se usa para 'hx' (como arriba) o aca como hacerlo con una clase

agricultura.find_all(attrs={'class':'dataset-title'})

agricultura.find_all('h3', attrs={'class':'dataset-title'})

#%%

agricultura.find_all('li', attrs={'class','filter-value'})

#%%

print(agricultura.find('li').get('style'))

#%%

div = agricultura.find(attrs={'class':'dataset-title'})
div.get('class')

#%%
    
agricultura.find_all('h3', attrs={'class', 'dataset-title'})    

#%%

divs = agricultura.find_all(attrs={'class', 'dataset-title'})
len(divs)

#for row in divs:
    
#%%

div = agricultura.find(attrs={'class':'dataset-title'})
print (div)

#%%

rows = div.find_all('tr') #tr: table row
for row in rows:
  celdas = row.find_all('td')
  print(celdas[0].text, celdas[1].text, celdas[2].text)
