import pandas as pd

# 1. Leer la versión anterior del dataset (v2)
df = pd.read_csv("data/dataset_v2.csv")

# 2. Verificar qué columnas tiene
print("Columnas del dataset v2:", list(df.columns))

# 3. Crear nuevas variables derivadas si existen las columnas esperadas
df = df.copy()

if "AveRooms" in df.columns and "AveOccup" in df.columns:
    # Densidad de habitaciones por ocupante
    df["RoomsPerOccupant"] = df["AveRooms"] / df["AveOccup"]
else:
    print("No se encontraron columnas AveRooms y AveOccup. Se omite RoomsPerOccupant.")

if "AveBedrms" in df.columns and "AveRooms" in df.columns:
    # Relación dormitorios por habitación
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]
else:
    print("No se encontraron columnas AveBedrms y AveRooms. Se omite BedroomsPerRoom.")

# 4. Reemplazar valores infinitos o NaN por la media
df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
df = df.fillna(df.mean(numeric_only=True))

# 5. Guardar dataset v3
print("Columnas nuevas agregadas (si aplican): RoomsPerOccupant, BedroomsPerRoom")
print("Nueva forma del dataset v3:", df.shape)

df.to_csv("data/dataset_v3.csv", index=False)
