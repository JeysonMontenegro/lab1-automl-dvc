import numpy as np
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import joblib


def main():
    # Cargar datos procesados
    data_npz = np.load("data/processed.npz")
    X_test = data_npz["X_test"]
    y_test = data_npz["y_test"]

    # Cargar mejor modelo entrenado
    best_model = joblib.load("models/best_model.joblib")

    # Predicciones
    y_pred = best_model.predict(X_test)

    # Métricas globales del mejor modelo
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Cargar resultados de todos los modelos para saber cuál fue el mejor
    models_results_path = Path("models/models_results.json")
    best_name = None
    best_r2 = None
    best_mse = None

    if models_results_path.exists():
        with open(models_results_path, "r") as f:
            all_models = json.load(f)

        # Elegimos el mejor según R2
        best = max(all_models, key=lambda m: m["r2_score"])
        best_name = best["name"]
        best_r2 = best["r2_score"]
        best_mse = best["mse"]
    else:
        all_models = []
        best_name = "unknown"
        best_r2 = float(r2)
        best_mse = float(mse)

    metrics = {
        "best_model": best_name,
        "best_model_r2": float(best_r2),
        "best_model_mse": float(best_mse),
        "r2_score": float(r2),
        "mse": float(mse),
        "all_models": all_models
    }

    # Guardar metrics.json (archivo que usa DVC)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
