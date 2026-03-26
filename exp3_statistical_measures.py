# ============================================================
# Experiment 3: Statistical Descriptions and Measures of
#               Similarity and Dissimilarity
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, cityblock, cosine
import matplotlib.pyplot as plt
import seaborn as sns
warnings._filters_mutated = lambda: None
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 3: STATISTICAL DESCRIPTIONS AND")
print("  MEASURES OF SIMILARITY & DISSIMILARITY")
print("=" * 60)

# Load Iris Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['SepalLength','SepalWidth','PetalLength','PetalWidth'])
df['Class'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(f"\nDataset: Iris")
print(f"Instances: {df.shape[0]}  |  Attributes: {df.shape[1]}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================
# STEP 1: STATISTICAL DESCRIPTION
# ============================================================
print("\n" + "=" * 60)
print("  STEP 1: STATISTICAL DESCRIPTION")
print("=" * 60)

numeric_cols = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

for col in numeric_cols:
    print(f"\n--- {col} ---")
    print(f"  Minimum        : {df[col].min():.3f}")
    print(f"  Maximum        : {df[col].max():.3f}")
    print(f"  Mean           : {df[col].mean():.3f}")
    print(f"  Median         : {df[col].median():.3f}")
    print(f"  Std Deviation  : {df[col].std():.3f}")
    print(f"  Variance       : {df[col].var():.3f}")
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    print(f"  Q1 (25%)       : {Q1:.3f}")
    print(f"  Q3 (75%)       : {Q3:.3f}")
    print(f"  IQR            : {Q3 - Q1:.3f}")

print("\n--- Full Describe() Summary ---")
print(df[numeric_cols].describe().round(3))

# ============================================================
# STEP 2: MEASURES OF SIMILARITY
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: MEASURES OF SIMILARITY")
print("=" * 60)

# Take sample points from each class
setosa     = df[df['Class'] == 'setosa'][numeric_cols].iloc[0].values
versicolor = df[df['Class'] == 'versicolor'][numeric_cols].iloc[0].values
virginica  = df[df['Class'] == 'virginica'][numeric_cols].iloc[0].values

print(f"\nSample Points:")
print(f"  Setosa     : {setosa}")
print(f"  Versicolor : {versicolor}")
print(f"  Virginica  : {virginica}")

# Euclidean Distance
print("\n[Euclidean Distance] (Lower = More Similar)")
print(f"  Setosa vs Versicolor : {euclidean(setosa, versicolor):.4f}")
print(f"  Setosa vs Virginica  : {euclidean(setosa, virginica):.4f}")
print(f"  Versicolor vs Virginica: {euclidean(versicolor, virginica):.4f}")

# Manhattan Distance
print("\n[Manhattan Distance] (Lower = More Similar)")
print(f"  Setosa vs Versicolor : {cityblock(setosa, versicolor):.4f}")
print(f"  Setosa vs Virginica  : {cityblock(setosa, virginica):.4f}")
print(f"  Versicolor vs Virginica: {cityblock(versicolor, virginica):.4f}")

# Cosine Similarity
print("\n[Cosine Similarity] (Higher = More Similar, Max=1)")
cos_sv  = 1 - cosine(setosa, versicolor)
cos_svi = 1 - cosine(setosa, virginica)
cos_vvi = 1 - cosine(versicolor, virginica)
print(f"  Setosa vs Versicolor : {cos_sv:.4f}")
print(f"  Setosa vs Virginica  : {cos_svi:.4f}")
print(f"  Versicolor vs Virginica: {cos_vvi:.4f}")

# ============================================================
# STEP 3: MEASURES OF DISSIMILARITY
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: MEASURES OF DISSIMILARITY")
print("=" * 60)

# Pairwise Distance Matrix for first 6 instances
sample = df[numeric_cols].head(6).values
print("\nPairwise Euclidean Distance Matrix (first 6 instances):")
dist_matrix = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        dist_matrix[i][j] = round(euclidean(sample[i], sample[j]), 3)

dist_df = pd.DataFrame(dist_matrix,
                        index=[f'P{i+1}' for i in range(6)],
                        columns=[f'P{i+1}' for i in range(6)])
print(dist_df)

print("\nInterpretation:")
print("  - Values close to 0    → High SIMILARITY")
print("  - Large values          → High DISSIMILARITY")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Experiment 3: Statistical Description & Similarity Measures', fontsize=14, fontweight='bold')

# Box plots
df[numeric_cols].boxplot(ax=axes[0][0])
axes[0][0].set_title('Box Plots - Statistical Description')
axes[0][0].set_ylabel('Value')

# Correlation Heatmap
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0][1])
axes[0][1].set_title('Correlation Matrix (Similarity)')

# Distance Bar Chart
labels = ['Set vs Ver', 'Set vs Vir', 'Ver vs Vir']
euc = [euclidean(setosa, versicolor), euclidean(setosa, virginica), euclidean(versicolor, virginica)]
man = [cityblock(setosa, versicolor), cityblock(setosa, virginica), cityblock(versicolor, virginica)]
x = np.arange(len(labels))
w = 0.35
axes[1][0].bar(x - w/2, euc, w, label='Euclidean', color='steelblue')
axes[1][0].bar(x + w/2, man, w, label='Manhattan', color='orange')
axes[1][0].set_title('Dissimilarity: Distance Measures')
axes[1][0].set_xticks(x)
axes[1][0].set_xticklabels(labels)
axes[1][0].legend()
axes[1][0].set_ylabel('Distance')

# Distance Matrix Heatmap
sns.heatmap(dist_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1][1])
axes[1][1].set_title('Pairwise Distance Matrix (Dissimilarity)')

plt.tight_layout()
plt.savefig('exp3_statistical_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'exp3_statistical_output.png'")

print("\n" + "=" * 60)
print("  EXPERIMENT 3 COMPLETED SUCCESSFULLY")
print("=" * 60)
