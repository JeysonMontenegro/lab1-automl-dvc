import numpy as np
import yaml
import json
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib


def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(name: str, cfg: dict):
    """
    Construye un modelo de sklearn según la configuración definida en params.yaml
    """
    model_type = cfg["type"]

    if model_type == "linear_regression":
        return LinearRegression()

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth"),
            random_state=cfg.get("random_state", 42),
        )

    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=cfg.get("n_estimators", 100),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth", 3),
            random_state=cfg.get("random_state", 42),
        )

    raise ValueError(f"Modelo no soportado: {name} ({model_type})")


def main():
    # 1. Cargar parámetros
    params = load_params()

    # 2. Cargar datos procesados
    data_npz = np.load("data/processed.npz")
    X_train = data_npz["X_train"]
    X_test = data_npz["X_test"]
    y_train = data_npz["y_train"]
    y_test = data_npz["y_test"]

    models_cfg = params["models"]

    results = []
    best_model = None
    best_name = None
    best_r2 = -999
    best_mse = None

    # 3. Entrenar y evaluar cada modelo definido en params.yaml
    for name, cfg in models_cfg.items():
        model = build_model(name, cfg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results.append({
            "name": name,
            "type": cfg["type"],
            "r2_score": float(round(r2, 5)),
            "mse": float(round(mse, 5))
        })

        if r2 > best_r2:
            best_r2 = r2
            best_mse = mse
            best_model = model
            best_name = name

    # 4. Guardar el mejor modelo
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")

    # 5. Guardar resultados de todos los modelos
    with open("models/models_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Mejor modelo: {best_name} (R2={best_r2:.4f}, MSE={best_mse:.4f})")


if __name__ == "__main__":
    main()
