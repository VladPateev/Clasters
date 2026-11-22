import sys
import pickle
import pandas as pd

# загрузка модели
with open("kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

# чтение данных для предсказания
path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
data = pd.read_csv(path)

cols = [c for c in data.columns if c.startswith("feature_")]
if not cols:
    cols = [c for c in data.columns if c != "cluster"]

# предсказание кластеров
predictions = model.predict(data[cols].values)
data["predicted_cluster"] = predictions

# сохранение результатов
data.to_csv("predictions.csv", index=False)
print(f"Предсказания сохранены в predictions.csv ({len(predictions)} строк)")

if "cluster" in data.columns:
    from sklearn.metrics import adjusted_rand_score
    print("Метрика ARI:", adjusted_rand_score(data["cluster"], predictions))

print(data[["predicted_cluster"] + cols[:2]].head())
