import pandas as pd
import pathlib as pl

#%%

"abro el excel"
direccion_actual = pl.Path(__file__).parent.resolve()
str_dir = str(direccion_actual)

#%%

#pobla = pd.read_excel(str_dir+"/DB/padron_poblacion.xlsX")

pobla = pd.read_excel(str_dir+"/padron_poblacion.xlsX")

#%%

"elimino la primera columna que no me sirve "
poblac = pobla.iloc[:, 1:]

#%%

"elimino la ultima tabla que es el resumen que tampoco me servira (por ahora)"
poblacion = poblac.iloc[:56596]

#%%

"obtengo las posiciones donde aparece el patron AREA#...  que son los nombres de los depas"
areas = poblacion[poblacion.apply(lambda r: r.astype(str).str.contains(r"AREA\s*#\s*\d+", case=False, na=False).any(), axis=1)]

#%%

"los convierto en lista para poder manejarlo con en for"
indices = areas.index.tolist()

#%%

resultados = []

"Hago enumarate(indices) para tmb obtener un contador, y poder ver facilmente si queda una tabla mas por recorrer o no"
for i, start in enumerate(indices):
    "marco el final de la tabla como el inicio de una nueva(que es donde aparece AREA...)" 
    if i < len(indices) - 1:
        end = indices[i + 1]
        "Y si estoy en la ultima tabla marco como el final el final del excel(ya que elimine el resumen)"
    else:
        end = len(poblacion)
    
    "Aca me separo el bloque con el que trabajare y le reinicio los indices asi es mas facil su manipulacion"    
    bloque = poblacion.loc[start+1:end-1].reset_index(drop=True)
    
    " nombre del departamento (segunda columna de la fila AREA... )"
    depto = poblacion.iloc[start, 1]
    
    "Toda la columna edades y casos las convierto en int, porque puede ser que se vea como numeros pero son str(que es lo que pasa)"
    bloque["Unnamed: 1"] = pd.to_numeric(bloque["Unnamed: 1"], errors='coerce')
    bloque["Unnamed: 2"] = pd.to_numeric(bloque["Unnamed: 2"], errors='coerce')
    
    "elimino filas que tienen como valor nulls, por ej arriba converti la fila con el total en nulls entonces aca la estoy eliminando y solo me quedo con las edades validas"
    bloque = bloque.dropna(subset=["Unnamed: 1", "Unnamed: 2"])
    

    "Aca filtro el dataframe BLOQUE, que seria la tabla del departamento x, donde me quedo con las filas que cumplen el rango que yo quiero y las sumo"
    jardin = bloque[(bloque["Unnamed: 1"] >= 3) & (bloque["Unnamed: 1"] <= 5)]["Unnamed: 2"].sum()
    primaria = bloque[(bloque["Unnamed: 1"] >= 6) & (bloque["Unnamed: 1"] <= 12)]["Unnamed: 2"].sum()
    secundaria = bloque[(bloque["Unnamed: 1"] >= 13) & (bloque["Unnamed: 1"] <= 18)]["Unnamed: 2"].sum()
    "Aca directamente sumo todas las filas para obtener el total"
    total = bloque["Unnamed: 2"].sum()
    

    "Con los datos obtenidos me creo un diccionario, los agrego a la lista ya creada antes"
    resultados.append(
         {"Departamento": depto,
          "Jardín": jardin,
          "Primaria": primaria,
          "Secundaria": secundaria,
          "Población Total": total}
        )

"al finalizar el for obtengo una lista de diccionarios, solo queda convertirlo a DataFrame"
df_final = pd.DataFrame(resultados) 