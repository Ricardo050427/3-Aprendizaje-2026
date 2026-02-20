import random
from collections import Counter
import arboles_numericos as an


def entrena_bosque(datos, target, clase_default, m_subconjuntos=10, variables_por_nodo=None, **kwargs):
    """
    Entrena un bosque aleatorio (Random Forest).

    Parámetros:
    -----------
    datos: list(dict)
        El conjunto de datos de entrenamiento.
    target: str
        El nombre de la columna objetivo.
    clase_default: str
        Valor por defecto para la clase.
    m_subconjuntos: int
        Número de árboles a entrenar (M).
    variables_por_nodo: int
        Cantidad de variables aleatorias a considerar en cada división.
    **kwargs:
        Argumentos adicionales para entrena_arbol (max_profundidad, acc_nodo, etc.).
    """
    bosque = []

    for _ in range(m_subconjuntos):
        subconjunto = random.choices(datos, k=len(datos))

        arbol = an.entrena_arbol(
            subconjunto,
            target,
            clase_default,
            variables_seleccionadas=variables_por_nodo,
            **kwargs
        )
        bosque.append(arbol)

    return bosque


def predice_instancia_bosque(bosque, instancia):
    """
    Obtiene la predicción de una sola instancia mediante el voto de la mayoría.
    """
    predicciones = [arbol.predice(instancia) for arbol in bosque]

    conteo = Counter(predicciones)
    return conteo.most_common(1)[0][0]


def predice_bosque(bosque, datos):
    """
    Realiza predicciones para una lista de instancias.
    """
    return [predice_instancia_bosque(bosque, d) for d in datos]


def evalua_bosque(bosque, datos, target):
    """
    Calcula la precisión del bosque en un conjunto de datos.
    """
    predicciones = predice_bosque(bosque, datos)
    aciertos = sum(1 for p, d in zip(predicciones, datos) if p == d[target])
    return aciertos / len(datos)