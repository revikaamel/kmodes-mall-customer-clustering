# ============================================================
# K-MEANS CLUSTERING - DATASET 1 (Mall Customers)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("data/Mall_Customers.csv")
print("Dataset Loaded:", df.shape)
print(df.head())

# ============================================================
# Preprocessing
# ============================================================

df_clean = df.drop(columns=["CustomerID"])
df_clean["Gender"] = df_clean["Gender"].map({"Male": 0, "Female": 1})

scaler = StandardScaler()
scaled = scaler.fit_transform(df_clean)
scaled_df = pd.DataFrame(scaled, columns=df_clean.columns)

# ============================================================
# Elbow Method
# ============================================================

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_df)
    wcss.append(km.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig("output/elbow_plot.png")
plt.close()

# ============================================================
# Silhouette Score
# ============================================================

sil_scores = []
for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_df)
    sil = silhouette_score(scaled_df, labels)
    sil_scores.append(sil)
    print(f"k={k}, Silhouette Score={sil:.4f}")

plt.figure(figsize=(7,5))
plt.plot(range(2,11), sil_scores, marker="o")
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.savefig("output/silhouette_plot.png")
plt.close()

# ============================================================
# Train Final Model
# ============================================================

best_k = 5  # kamu bisa ubah sesuai hasil grafik
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = model.fit_predict(scaled_df)

df["Cluster"] = cluster_labels
print("\nCluster Added:")
print(df.head())

# ============================================================
# Visualisasi Hasil Cluster
# ============================================================

plt.figure(figsize=(7,5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=60
)
plt.title("K-Means Clustering Result")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.savefig("output/cluster_scatter.png")
plt.close()

print("\nAll visuals saved in /output folder.")
