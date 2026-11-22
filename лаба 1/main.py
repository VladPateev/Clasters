import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

# загрузка датасета
df = pd.read_csv("dataset.csv")
print("Размер датасета:", df.shape)

feature_cols = [c for c in df.columns if c.startswith("feature_")]
if not feature_cols:
    feature_cols = list(df.columns)

X = df[feature_cols].values

# поиск оптимального k методом локтя
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    print(f"k = {k}: инерция = {kmeans.inertia_:.1f}")

# код для создания графиков метода локтя
plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, 'bo-', linewidth=2)
plt.axvline(5, color='r', linestyle='--', label='Оптимальное k=5')
plt.title("Метод локтя для поиска числа кластеров")
plt.xlabel("Число кластеров k")
plt.ylabel("Inertia (WCSS)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("elbow.png")
plt.close()

# обучение модели k-means
optimal_k = 5
model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = model.fit_predict(X)

# сохранение модели и разметки
df["cluster"] = labels
df.to_csv("clustered_dataset.csv", index=False)

with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(model, f)

# код для создания графиков визуализации кластеров
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(model.cluster_centers_)

plt.figure(figsize=(9, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=2, alpha=0.4)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=150, linewidths=2, label='Центроиды')
for i, (cx, cy) in enumerate(centers_pca):
    plt.annotate(f"Кластер {i}", (cx, cy), xytext=(5, 5), textcoords='offset points', weight='bold')

plt.title("Кластеризация K-Means (k=5) в 2D через PCA")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("clusters.png")
plt.close()

print("\nРаспределение точек по кластерам:")
print(df["cluster"].value_counts().sort_index())
print("\nКластеризация завершена. Файлы сохранены.")
