# ============================================================
# Experiment 9: K-Means Clustering (SimpleKMeans)
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 9: K-MEANS CLUSTERING (SimpleKMeans)")
print("=" * 60)

# ============================================================
# DATASET - Iris (equivalent to Soybean in concept)
# ============================================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['SepalLength','SepalWidth','PetalLength','PetalWidth'])
df['TrueClass'] = iris.target_names[iris.target]

print("\nDataset: Iris")
print(f"Instances  : {len(df)}")
print(f"Attributes : {df.shape[1] - 1} (numeric)")
print(f"\nFirst 5 Rows:")
print(df.head())
print(f"\nClass Distribution:\n{df['TrueClass'].value_counts()}")

# ============================================================
# STEP 1: DATA PREPARATION
# ============================================================
print("\n[STEP 1] DATA PREPARATION")
print("-" * 50)

X = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data standardized (mean=0, std=1)")

# ============================================================
# STEP 2: FIND OPTIMAL K (Elbow Method)
# ============================================================
print("\n[STEP 2] FINDING OPTIMAL K (Elbow Method)")
print("-" * 50)

inertia_vals = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia_vals.append(km.inertia_)
    print(f"  K={k}: Inertia={km.inertia_:.4f}")

# ============================================================
# STEP 3: APPLY K-MEANS WITH K=3
# ============================================================
print("\n[STEP 3] APPLYING K-MEANS WITH K=3 (Euclidean Distance)")
print("-" * 50)

K = 3
kmeans = KMeans(n_clusters=K, init='random', random_state=42, n_init=10, max_iter=500)
kmeans.fit(X_scaled)
labels = kmeans.labels_

df['Cluster'] = labels

print(f"\nNumber of Iterations   : {kmeans.n_iter_}")
print(f"Sum of Squared Errors  : {kmeans.inertia_:.4f}")

# ============================================================
# STEP 4: CLUSTER CENTROIDS
# ============================================================
print("\n[STEP 4] CLUSTER CENTROIDS")
print("-" * 50)

centroids_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=['SepalLength','SepalWidth','PetalLength','PetalWidth']
)
centroids_df.index = [f'Cluster {i}' for i in range(K)]
print(centroids_df.round(4))

# ============================================================
# STEP 5: CLUSTER DISTRIBUTION
# ============================================================
print("\n[STEP 5] CLUSTER DISTRIBUTION")
print("-" * 50)

for i in range(K):
    count = np.sum(labels == i)
    pct = count / len(df) * 100
    print(f"  Cluster {i}: {count:3d} instances ({pct:.1f}%)")

# ============================================================
# STEP 6: CLUSTER EVALUATION METRICS
# ============================================================
print("\n[STEP 6] CLUSTER EVALUATION METRICS")
print("-" * 50)

sil  = silhouette_score(X_scaled, labels)
db   = davies_bouldin_score(X_scaled, labels)
ch   = calinski_harabasz_score(X_scaled, labels)

print(f"  Silhouette Score        : {sil:.4f}  (higher = better, max=1)")
print(f"  Davies-Bouldin Index    : {db:.4f}  (lower = better)")
print(f"  Calinski-Harabasz Score : {ch:.4f}  (higher = better)")

# ============================================================
# STEP 7: SIMILARITY INTERPRETATION
# ============================================================
print("\n[STEP 7] SIMILARITY & DISSIMILARITY INTERPRETATION")
print("-" * 50)

for i in range(K):
    cluster_data = X[labels == i]
    mean = cluster_data.mean(axis=0)
    std  = cluster_data.std(axis=0)
    print(f"\n  Cluster {i}:")
    print(f"    Mean PetalLength = {mean[2]:.3f}  |  Std = {std[2]:.3f}")
    print(f"    Mean PetalWidth  = {mean[3]:.3f}  |  Std = {std[3]:.3f}")

print("\n  → Instances WITHIN the same cluster → HIGH SIMILARITY (close Euclidean distance to centroid)")
print("  → Instances in DIFFERENT clusters   → HIGH DISSIMILARITY (large Euclidean distance)")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle('Experiment 9: K-Means Clustering', fontsize=14, fontweight='bold')

colors = ['#e74c3c', '#3498db', '#2ecc71']

# Elbow Curve
axes[0][0].plot(list(k_range), inertia_vals, 'bo-', linewidth=2, markersize=6)
axes[0][0].axvline(x=3, color='red', linestyle='--', label='K=3 (Optimal)')
axes[0][0].set_title('Elbow Method - Finding Optimal K')
axes[0][0].set_xlabel('Number of Clusters (K)')
axes[0][0].set_ylabel('Inertia (SSE)')
axes[0][0].legend()
axes[0][0].grid(True, alpha=0.3)

# Scatter: SepalLength vs PetalLength
for i in range(K):
    mask = labels == i
    axes[0][1].scatter(df['SepalLength'][mask], df['PetalLength'][mask],
                       c=colors[i], label=f'Cluster {i}', s=60, edgecolors='black', linewidth=0.5)
axes[0][1].scatter(centroids_df['SepalLength'], centroids_df['PetalLength'],
                   c='black', marker='X', s=200, zorder=5, label='Centroids')
axes[0][1].set_title('Clusters: SepalLength vs PetalLength')
axes[0][1].set_xlabel('SepalLength')
axes[0][1].set_ylabel('PetalLength')
axes[0][1].legend()

# Scatter: PetalLength vs PetalWidth
for i in range(K):
    mask = labels == i
    axes[1][0].scatter(df['PetalLength'][mask], df['PetalWidth'][mask],
                       c=colors[i], label=f'Cluster {i}', s=60, edgecolors='black', linewidth=0.5)
axes[1][0].scatter(centroids_df['PetalLength'], centroids_df['PetalWidth'],
                   c='black', marker='X', s=200, zorder=5, label='Centroids')
axes[1][0].set_title('Clusters: PetalLength vs PetalWidth (Key Features)')
axes[1][0].set_xlabel('PetalLength')
axes[1][0].set_ylabel('PetalWidth')
axes[1][0].legend()

# Cluster distribution bar chart
counts = [np.sum(labels == i) for i in range(K)]
axes[1][1].bar([f'Cluster {i}' for i in range(K)], counts, color=colors, edgecolor='black')
axes[1][1].set_title('Cluster Distribution (Instance Count)')
axes[1][1].set_ylabel('Number of Instances')
for i, v in enumerate(counts):
    axes[1][1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('exp9_kmeans_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'exp9_kmeans_output.png'")

print("\n" + "=" * 60)
print("  EXPERIMENT 9 COMPLETED SUCCESSFULLY")
print("=" * 60)
