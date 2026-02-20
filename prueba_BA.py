import utileria as ut
import arboles_numericos as an
import bosque_aleatorio as ba
import os
import random

# 1.- Preparación de datos
url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
if not os.path.exists("datos"): os.makedirs("datos")
archivo, archivo_datos = "datos/cancer.zip", "datos/wdbc.data"
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)
    ut.descomprime_zip(archivo)

atributos_nombres = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
datos = ut.lee_csv(archivo_datos, atributos=atributos_nombres)

for d in datos:
    d['Diagnosis'] = 1 if d['Diagnosis'] == 'M' else 0
    for i in range(1, 31): d[f'feature_{i}'] = float(d[f'feature_{i}'])
    del(d['ID'])

target = 'Diagnosis'
random.seed(42)
random.shuffle(datos)
N = int(0.8 * len(datos))
train, val = datos[:N], datos[N:]

# 2.- Árbol Único vs Bosque
print("Variando el número de árboles".center(40))
print("-" * 40)

# Árbol clásico
arbol_unico = an.entrena_arbol(train, target, clase_default=0, max_profundidad=5)
acc_arbol = an.evalua_arbol(arbol_unico, val, target)
print(f"Accuracy Árbol Único: {acc_arbol:.4f}")

# Bosque Aleatorio
# k = sqrt(30 variables) aproximadamente 5 o 6
for m in [10, 50, 100]:
    bosque = ba.entrena_bosque(
        train, target, clase_default=0,
        m_subconjuntos=m,
        variables_por_nodo=5,
        max_profundidad=5
    )
    acc_bosque = ba.evalua_bosque(bosque, val, target)
    print(f"Accuracy Bosque (M={m}): {acc_bosque:.4f}")

print("\nVariando la profundidad máxima")
print("-" * 40)
for prof in [2, 5, 10, None]:
    bosque = ba.entrena_bosque(
        train, target, clase_default=0,
        m_subconjuntos=50, variables_por_nodo=5, max_profundidad=prof
    )
    acc_bosque = ba.evalua_bosque(bosque, val, target)
    print(f"Accuracy Bosque (Profundidad={prof}): {acc_bosque:.4f}")

print("\nVariando las variables por nodo")
print("-" * 40)
for var in [2, 5, 10, 20, 30]:
    bosque = ba.entrena_bosque(
        train, target, clase_default=0,
        m_subconjuntos=50, variables_por_nodo=var, max_profundidad=5
    )
    acc_bosque = ba.evalua_bosque(bosque, val, target)
    print(f"Accuracy Bosque (Variables={var}): {acc_bosque:.4f}")
