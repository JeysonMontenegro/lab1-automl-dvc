import pandas as pd

# 1. Leo la versión anterior del dataset (v3)
df = pd.read_csv("data/dataset_v3.csv")

print("Forma original de v3:", df.shape)

# 2. Filtro casas con precio medio 
df_v4 = df[df["MedHouseVal"] <= 4.0].copy()

print("Forma de v4 (solo precios <= 4.0):", df_v4.shape)

# 3. Guardo la nueva versión del dataset
df_v4.to_csv("data/dataset_v4.csv", index=False)
print("Guardado data/dataset_v4.csv")
