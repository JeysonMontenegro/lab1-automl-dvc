import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import yaml
import joblib
from pathlib import Path


def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    raw_path = params["data"]["raw_path"]
    target_col = params["data"]["target"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    # Cargar datos
    df = pd.read_csv(raw_path)

    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identificar tipos de columnas
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Definir transformadores
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Ajustar preprocesador y transformar
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Asegurar carpetas de salida
    data_dir = Path("data")
    models_dir = Path("models")
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    # Guardar datos procesados en un solo archivo .npz
    np.savez(
        data_dir / "processed.npz",
        X_train=X_train_prep,
        X_test=X_test_prep,
        y_train=y_train.values,
        y_test=y_test.values,
    )

    # Guardar el preprocesador para usos futuros
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")


if __name__ == "__main__":
    main()
