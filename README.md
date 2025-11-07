# Laboratorio 1 – AutoML y DVC

Este proyecto aplica **DVC** y **Git** para versionar datasets y automatizar un flujo de entrenamiento y evaluación de modelos tipo AutoML.  
El objetivo es observar cómo cambian las métricas de desempeño al modificar el conjunto de datos en diferentes versiones.

---

## Evolución del dataset

| Versión | Descripción breve |
|----------|------------------|
| **v1** | Dataset original (base inicial) |
| **v2** | Limpieza de outliers (poblaciones altas) |
| **v3** | Nuevas columnas derivadas (RoomsPerOccupant, BedroomsPerRoom) |
| **v4** | Filtro de viviendas con valor medio (MedHouseVal ≤ 4.0) |

---

## Resultados de métricas

| Versión | MSE | R² | Comentario |
|----------|-----|----|------------|
| v1 | 0.2615 | 0.8004 | Línea base |
| v2 | 0.2474 | 0.8146 | Mejora leve por limpieza |
| v3 | 0.2411 | 0.8193 | Mejora ligera por nuevas variables |
| v4 | 0.1694 | 0.7713 | Menor error pero menos generalización |

---

## Conclusión

Las versiones v2 y v3 mostraron mejoras graduales en el desempeño del modelo gracias a la limpieza y enriquecimiento de datos.  
La versión v4 redujo el error absoluto (MSE) al enfocarse en precios medios, pero también disminuyó el R², lo que indica una pérdida de capacidad de generalización.  
En conjunto, el ejercicio demuestra la importancia de los datos en el rendimiento de los modelos y cómo DVC facilita comparar, versionar y reproducir los experimentos.

---
