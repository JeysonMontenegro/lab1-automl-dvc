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

    # Métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    metrics = {
        "r2_score": float(r2),
        "mse": float(mse),
    }

    # Guardar como metrics.json
    metrics_path = Path("metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
