#%%

empleado_01 = [[20222333,45,2,20000],[33456234,40,0,25000],[45432345,41,1,10000]]

def superanSalarioActividad01(a : [int], umbral: int) -> [int] :
    res : [int] = []
    for i in a:
        if i[3] > umbral :
            res.append(i)
    return res

print(superanSalarioActividad01(empleado_01,15000))

#%%
        