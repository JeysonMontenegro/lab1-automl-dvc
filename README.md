# Laboratorio 1 – AutoML y DVC

**Universidad Galileo – Maestría en Data Science*  
**Curso:** Product Development
**Autor:** Jeyson Steve Montenegro Alay  
**Año:** 2025

---

## 1. Descripción general

Este laboratorio implementa un pipeline reproducible utilizando **DVC** y **Git** para gestionar datasets, entrenar modelos y automatizar la comparación de resultados (AutoML sencillo).  
El objetivo fue observar cómo los cambios en los datos y configuraciones afectan el rendimiento del modelo de predicción del valor medio de vivienda (*MedHouseVal*).

El pipeline tiene tres etapas principales:

1. **Preprocess:** limpieza, codificación y división de los datos.  
2. **Train:** entrenamiento de modelos configurados en `params.yaml`.  
3. **Evaluate:** evaluación del mejor modelo y registro de métricas en `metrics.json`.

---

## 2. Dependencias y configuración del entorno

El proyecto fue desarrollado en **Windows 11**, con **Python 3.11** y un entorno administrado por **Conda**.

### Librerías necesarias


```bash
 dvc scikit-learn pandas numpy pyyaml joblib
```

También es necesario tener instalado **Git** para el control de versiones.

### Estructura del proyecto

```text
lab1-automl-dvc/
│
├── data/
│   ├── dataset_v1.csv
│   ├── dataset_v2.csv
│   ├── dataset_v3.csv
│   └── dataset_v4.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── params.yaml
├── dvc.yaml
├── metrics.json
├── report.md
└── README.md
```

---

## 3. Instrucciones de ejecución

Para reproducir el pipeline completo y comparar resultados:

### Clonar o abrir el proyecto

```bash
git clone <repositorio>
cd lab1-automl-dvc
```

### Ejecutar el pipeline

```bash
dvc repro
```

Esto ejecutará automáticamente las etapas **preprocess**, **train** y **evaluate**, generando los archivos de salida y métricas.

### Ver resultados

```bash
dvc metrics show
```

### Comparar versiones

```bash
dvc metrics diff HEAD~1
```

### Cambiar dataset o configuración

Modificar la ruta del dataset en `params.yaml`:

```yaml
data:
  raw_path: "data/dataset_v4.csv"
```

Luego volver a ejecutar:

```bash
dvc repro
```

---

## 4. Evolución de Datos y Versionado

Durante el desarrollo del laboratorio se crearon y versionaron progresivamente cuatro versiones del dataset con **DVC** y **Git**, aplicando mejoras y transformaciones en cada una.  
Cada versión fue registrada con un commit descriptivo, y los resultados del modelo se compararon usando `dvc metrics diff`.

| Versión | Commit | Descripción del cambio | Tipo de modificación | Resultado principal |
|----------|---------|------------------------|----------------------|--------------------|
| **v1** | `938d612` | Dataset original versionado con DVC. | Base inicial | MSE: 0.2615 / R²: 0.8004 |
| **v2** | `40b4b4e` | Limpieza de outliers por población. | Limpieza de datos | MSE: 0.2474 / R²: 0.8146 |
| **v3** | `7ef1d0d` | Creación de variables derivadas (RoomsPerOccupant, BedroomsPerRoom). | Transformación / Feature engineering | MSE: 0.2411 / R²: 0.8193 |
| **v4** | `bbea521` | Filtro de viviendas con valor medio (MedHouseVal ≤ 4.0). | Reducción / Segmentación de datos | MSE: 0.1694 / R²: 0.7713 |

---

## 5. Resultados y Discusión

El análisis completo de resultados, comparación de métricas y conclusiones se encuentra en el archivo [`report.md`](./report.md).

De forma resumida:
- Las versiones **v2** y **v3** mejoraron el rendimiento del modelo gracias a la limpieza y al enriquecimiento de variables.
- La versión **v4** redujo el error absoluto (MSE) pero también la capacidad de generalización (R²).

---

## 6. Conclusiones

- La calidad de los datos influye directamente en el rendimiento del modelo.  
- La limpieza y creación de variables aportaron mejoras graduales en el desempeño.  
- Limitar el rango de los datos redujo el error absoluto, pero también la capacidad de generalización.  
- DVC permitió mantener trazabilidad, reproducibilidad y comparación entre versiones.  
- El componente AutoML seleccionó automáticamente el mejor modelo sin intervención manual.

