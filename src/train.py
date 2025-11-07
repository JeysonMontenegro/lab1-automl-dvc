import numpy as np
import yaml
import json
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib


def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(name: str, cfg: dict):
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
    params = load_params()

    # Cargar datos procesados
    data_npz = np.load("data/processed.npz")
    X_train = data_npz["X_train"]
    X_test = data_npz["X_test"]
    y_train = data_npz["y_train"]
    y_test = data_npz["y_test"]

    models_cfg = params["models"]

    results = []
    best_model = None
    best_name = None
    best_score = -1e9  # Muy bajo para empezar

    for name, cfg in models_cfg.items():
        model = build_model(name, cfg)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

        results.append(
            {
                "name": name,
                "type": cfg["type"],
                "r2_score": float(score),
            }
        )

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Guardar el mejor modelo
    joblib.dump(best_model, models_dir / "best_model.joblib")

    # Guardar información de resultados para referencia
    with open(models_dir / "models_results.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "best_r2_score": float(best_score),
                "all_models": results,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
