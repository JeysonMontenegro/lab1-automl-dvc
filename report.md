# Reporte Final – Laboratorio 1: AutoML y DVC

## 1. Resumen del Experimento

Se construyó un pipeline reproducible con DVC que automatiza el preprocesamiento, entrenamiento y evaluación de modelos tipo AutoML.  
El objetivo fue analizar cómo las distintas versiones del dataset afectan el rendimiento del modelo de predicción del valor medio de vivienda (MedHouseVal).

El pipeline consta de tres etapas:
1. **Preprocess:** limpieza, codificación y división de los datos.
2. **Train:** entrenamiento automático de varios modelos definidos en `params.yaml`.
3. **Evaluate:** evaluación del mejor modelo y registro de métricas en `metrics.json`.

---

## 2. Versiones del Dataset

| Versión | Descripción |
|----------|-------------|
| v1 | Dataset original sin cambios. |
| v2 | Limpieza de outliers (poblaciones altas). |
| v3 | Nuevas variables derivadas (RoomsPerOccupant, BedroomsPerRoom). |
| v4 | Filtro de viviendas con valor medio (MedHouseVal ≤ 4.0). |

Cada versión se versionó con DVC y fue reproducida mediante `dvc repro`, permitiendo mantener un control total sobre los cambios y sus efectos en el modelo.

---

## 3. Resultados y Comparación

| Comparación | MSE Anterior | MSE Nuevo | Δ MSE | R² Anterior | R² Nuevo | Δ R² |
|--------------|--------------|------------|--------|--------------|-----------|-------|
| v1 → v2 | 0.2615 | 0.2474 | -0.0141 | 0.8004 | 0.8146 | +0.0142 |
| v2 → v3 | 0.2474 | 0.2411 | -0.0063 | 0.8146 | 0.8193 | +0.0047 |
| v3 → v4 | 0.2474 | 0.1694 | -0.0780 | 0.8146 | 0.7713 | -0.0433 |

Las versiones v2 y v3 mostraron mejoras graduales gracias a la limpieza y enriquecimiento de los datos.  
La versión v4 redujo el error absoluto (MSE), pero también la capacidad explicativa del modelo (R²).

---

## 4. Modelo Ganador y Parámetros Finales

Según `dvc metrics show`, el mejor modelo final fue **Gradient Boosting**, con:

- **R² final:** 0.7713  
- **MSE final:** 0.1694  

Configuración del modelo (definida en `params.yaml`):

```yaml
gradient_boosting:
  type: "gradient_boosting"
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 3
  random_state: 42
