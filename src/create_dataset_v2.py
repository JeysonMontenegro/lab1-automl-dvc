import pandas as pd

# Leer dataset v1
df = pd.read_csv("data/dataset_v1.csv")

# Ejemplo de "limpieza": eliminar barrios con población > 5000
df_v2 = df[df["Population"] <= 5000].copy()

print(f"Filas originales: {len(df)}")
print(f"Filas después de limpieza (v2): {len(df_v2)}")

# Guardar nueva versión del dataset
df_v2.to_csv("data/dataset_v2.csv", index=False)
