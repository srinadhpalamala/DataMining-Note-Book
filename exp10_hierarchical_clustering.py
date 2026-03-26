# ============================================================
# Experiment 10: Hierarchical Clustering
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import (dendrogram, linkage,
                                      fcluster, cophenet)
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 10: HIERARCHICAL CLUSTERING")
print("=" * 60)

# ============================================================
# DATASET - Glass dataset (using Iris as equivalent numeric data)
# ============================================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['SepalLength','SepalWidth','PetalLength','PetalWidth'])
df['TrueClass'] = iris.target_names[iris.target]

# Use subset for clearer dendrogram
df_small = df.sample(30, random_state=42).reset_index(drop=True)

print("\nDataset: Iris (30 samples for clear dendrogram)")
print(f"Instances  : {len(df_small)}")
print(f"Attributes : 4 (numeric)")
print(df_small.head())

# ============================================================
# STEP 1: DATA PREPARATION
# ============================================================
print("\n[STEP 1] DATA PREPARATION")
print("-" * 50)

X_full  = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']].values
X_small = df_small[['SepalLength','SepalWidth','PetalLength','PetalWidth']].values

scaler  = StandardScaler()
X_scaled_full  = scaler.fit_transform(X_full)
X_scaled_small = scaler.fit_transform(X_small)
print("Data standardized successfully.")

# ============================================================
# STEP 2: HIERARCHICAL CLUSTERING - LINKAGE METHODS
# ============================================================
print("\n[STEP 2] HIERARCHICAL CLUSTERING - LINKAGE METHODS")
print("-" * 50)

link_methods = ['single', 'complete', 'average', 'ward']
for method in link_methods:
    Z = linkage(X_scaled_full, method=method)
    c, coph_dist = cophenet(Z, pdist(X_scaled_full))
    print(f"  {method.capitalize():10s} Linkage - Cophenetic Correlation: {c:.4f}")

# ============================================================
# STEP 3: APPLY WITH AVERAGE LINKAGE (N=3 clusters)
# ============================================================
print("\n[STEP 3] APPLYING HIERARCHICAL CLUSTERING")
print("         Method: AVERAGE Linkage | Clusters: 3")
print("-" * 50)

Z_full = linkage(X_scaled_full, method='average')
labels = fcluster(Z_full, t=3, criterion='maxclust') - 1  # 0-indexed
df['Cluster'] = labels

print(f"\nCophenetic Correlation (Average): {cophenet(Z_full, pdist(X_scaled_full))[0]:.4f}")

# Cluster Distribution
print("\nCluster Distribution:")
for i in range(3):
    count = np.sum(labels == i)
    pct   = count / len(df) * 100
    print(f"  Cluster {i}: {count:3d} instances ({pct:.1f}%)")

# ============================================================
# STEP 4: CLUSTER CENTROIDS
# ============================================================
print("\n[STEP 4] CLUSTER CENTROIDS")
print("-" * 50)
for i in range(3):
    centroid = X_full[labels == i].mean(axis=0)
    print(f"  Cluster {i}: SepalLen={centroid[0]:.3f}, SepalWid={centroid[1]:.3f}, "
          f"PetalLen={centroid[2]:.3f}, PetalWid={centroid[3]:.3f}")

# ============================================================
# STEP 5: EVALUATION METRICS
# ============================================================
print("\n[STEP 5] EVALUATION METRICS")
print("-" * 50)

sil = silhouette_score(X_scaled_full, labels)
db  = davies_bouldin_score(X_scaled_full, labels)
print(f"  Silhouette Score     : {sil:.4f}  (higher = better)")
print(f"  Davies-Bouldin Index : {db:.4f}  (lower = better)")

# ============================================================
# STEP 6: COMPARE LINKAGE METHODS
# ============================================================
print("\n[STEP 6] COMPARISON OF LINKAGE METHODS")
print("-" * 50)
print(f"  {'Linkage':<12} {'Silhouette':>12} {'Davies-Bouldin':>16}")
print("  " + "-" * 42)
for method in link_methods:
    Z_m = linkage(X_scaled_full, method=method)
    l_m = fcluster(Z_m, t=3, criterion='maxclust') - 1
    try:
        sil_m = silhouette_score(X_scaled_full, l_m)
        db_m  = davies_bouldin_score(X_scaled_full, l_m)
        print(f"  {method.capitalize():<12} {sil_m:>12.4f} {db_m:>16.4f}")
    except Exception:
        print(f"  {method.capitalize():<12} {'N/A':>12}")

# ============================================================
# STEP 7: ANALYSIS
# ============================================================
print("\n[STEP 7] INTERPRETATION")
print("-" * 50)
print("""
  Hierarchical Clustering groups data by progressively merging
  the most similar instances/clusters:

  Linkage Types:
  - Single   : min distance between clusters (chain-like clusters)
  - Complete : max distance (compact, spherical clusters)
  - Average  : average distance (balanced approach)
  - Ward     : minimizes within-cluster variance (best for compact clusters)

  Results:
  → Instances within the same cluster have HIGH SIMILARITY
  → Instances in different clusters have HIGH DISSIMILARITY
  → The dendrogram shows the hierarchy of cluster merges
  → Cut the dendrogram at height to get desired number of clusters
""")

# ============================================================
# VISUALIZATION
# ============================================================
fig = plt.figure(figsize=(15, 12))
fig.suptitle('Experiment 10: Hierarchical Clustering', fontsize=14, fontweight='bold')

# Dendrogram (small dataset for clarity)
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
Z_small = linkage(X_scaled_small, method='average')
dendrogram(Z_small,
           labels=[f"S{i}" for i in range(len(X_scaled_small))],
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=1.5, ax=ax1)
ax1.set_title('Dendrogram (Average Linkage) - 30 Samples')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Distance')
ax1.axhline(y=1.5, color='red', linestyle='--', label='Cut-line (3 clusters)')
ax1.legend()

colors = ['#e74c3c', '#3498db', '#2ecc71']

# Scatter: SepalLength vs PetalLength
ax2 = plt.subplot2grid((2, 2), (1, 0))
for i in range(3):
    mask = labels == i
    ax2.scatter(df['SepalLength'][mask], df['PetalLength'][mask],
                c=colors[i], label=f'Cluster {i}', s=50, edgecolors='black', linewidth=0.5)
ax2.set_title('Clusters: SepalLength vs PetalLength')
ax2.set_xlabel('SepalLength')
ax2.set_ylabel('PetalLength')
ax2.legend()

# Scatter: PetalLength vs PetalWidth
ax3 = plt.subplot2grid((2, 2), (1, 1))
for i in range(3):
    mask = labels == i
    ax3.scatter(df['PetalLength'][mask], df['PetalWidth'][mask],
                c=colors[i], label=f'Cluster {i}', s=50, edgecolors='black', linewidth=0.5)
ax3.set_title('Clusters: PetalLength vs PetalWidth')
ax3.set_xlabel('PetalLength')
ax3.set_ylabel('PetalWidth')
ax3.legend()

plt.tight_layout()
plt.savefig('exp10_hierarchical_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'exp10_hierarchical_output.png'")

print("\n" + "=" * 60)
print("  EXPERIMENT 10 COMPLETED SUCCESSFULLY")
print("=" * 60)
