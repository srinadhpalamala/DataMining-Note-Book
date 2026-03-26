# ============================================================
# Experiment 4: Data Preprocessing on Weather Dataset
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 4: DATA PREPROCESSING - WEATHER DATASET")
print("=" * 60)

# ============================================================
# CREATE SAMPLE WEATHER DATASET
# ============================================================
np.random.seed(42)
n = 50

data = {
    'Pressure':          np.random.uniform(980, 1050, n),
    'Global_Radiation':  np.random.uniform(0, 1, n),
    'Temp_Mean':         np.random.uniform(-5, 40, n),
    'Temp_Min':          np.random.uniform(-10, 30, n),
    'Temp_Max':          np.random.uniform(0, 45, n),
    'Wind_Speed':        np.random.uniform(0, 30, n),
    'Wind_Bearing':      np.random.uniform(0, 360, n),
    'Label':             np.random.choice([0, 1], n)
}

# Inject missing values
df = pd.DataFrame(data)
for col in ['Pressure', 'Wind_Speed', 'Temp_Mean']:
    idx = np.random.choice(df.index, 5, replace=False)
    df.loc[idx, col] = np.nan

# Inject duplicates
df = pd.concat([df, df.iloc[:3]], ignore_index=True)

print("\nOriginal Dataset (first 10 rows):")
print(df.head(10))
print(f"\nShape: {df.shape}")

# ============================================================
# STEP 1: EXPLORE DATA
# ============================================================
print("\n[STEP 1] EXPLORATORY ANALYSIS")
print("-" * 50)
print(f"Total Records  : {len(df)}")
print(f"Total Attributes: {df.shape[1]}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicate Rows : {df.duplicated().sum()}")

# ============================================================
# STEP 2: REMOVE DUPLICATES
# ============================================================
print("\n[STEP 2] REMOVE DUPLICATES")
print("-" * 50)
before = len(df)
df = df.drop_duplicates()
print(f"Removed: {before - len(df)} duplicate rows | Remaining: {len(df)}")

# ============================================================
# STEP 3: HANDLE MISSING VALUES
# ============================================================
print("\n[STEP 3] HANDLE MISSING VALUES")
print("-" * 50)

# Fill numeric columns with mean
for col in df.columns:
    if df[col].isnull().sum() > 0:
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        print(f"  '{col}' missing values filled with mean: {mean_val:.4f}")

print(f"\nMissing values after handling:\n{df.isnull().sum()}")

# ============================================================
# STEP 4: REMOVE UNNECESSARY ATTRIBUTES
# ============================================================
print("\n[STEP 4] ATTRIBUTE SELECTION / REMOVAL")
print("-" * 50)
cols_to_remove = ['Wind_Bearing']
df = df.drop(columns=cols_to_remove)
print(f"Removed attributes: {cols_to_remove}")
print(f"Remaining attributes: {list(df.columns)}")

# ============================================================
# STEP 5: NORMALIZATION (Min-Max)
# ============================================================
print("\n[STEP 5] NORMALIZATION (Min-Max Scaling: 0 to 1)")
print("-" * 50)
features = ['Pressure', 'Global_Radiation', 'Temp_Mean', 'Temp_Min', 'Temp_Max', 'Wind_Speed']
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[features] = scaler.fit_transform(df[features])
print("Normalized Data (first 5 rows):")
print(df_normalized[features].head())

# ============================================================
# STEP 6: STANDARDIZATION (Z-Score)
# ============================================================
print("\n[STEP 6] STANDARDIZATION (Z-Score: mean=0, std=1)")
print("-" * 50)
std_scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[features] = std_scaler.fit_transform(df[features])
print("Standardized Data (first 5 rows):")
print(df_standardized[features].head())

# ============================================================
# STEP 7: OUTLIER DETECTION (IQR Method)
# ============================================================
print("\n[STEP 7] OUTLIER DETECTION (IQR Method)")
print("-" * 50)
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"  {col}: {len(outliers)} outlier(s) detected")

# ============================================================
# STEP 8: SAVE PREPROCESSED DATA
# ============================================================
df_normalized.to_csv('weather_preprocessed.csv', index=False)
print("\n[STEP 8] Preprocessed data saved as 'weather_preprocessed.csv'")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Experiment 4: Weather Data Preprocessing', fontsize=14, fontweight='bold')

# Before normalization
axes[0][0].boxplot([df[col].dropna() for col in features], labels=features)
axes[0][0].set_title('Before Normalization - Box Plot')
axes[0][0].tick_params(axis='x', rotation=30)

# After normalization
axes[0][1].boxplot([df_normalized[col] for col in features], labels=features)
axes[0][1].set_title('After Min-Max Normalization')
axes[0][1].tick_params(axis='x', rotation=30)

# After standardization
axes[1][0].boxplot([df_standardized[col] for col in features], labels=features)
axes[1][0].set_title('After Z-Score Standardization')
axes[1][0].tick_params(axis='x', rotation=30)

# Missing value heatmap (before)
miss_data = pd.DataFrame({col: [5 if col in ['Pressure','Wind_Speed','Temp_Mean'] else 0]
                           for col in df.columns})
axes[1][1].bar(df.columns, [5 if c in ['Pressure','Wind_Speed','Temp_Mean'] else 0
                              for c in df.columns], color='tomato')
axes[1][1].set_title('Missing Values (Before Handling)')
axes[1][1].set_ylabel('Count')
axes[1][1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('exp4_preprocessing_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved as 'exp4_preprocessing_output.png'")

print("\n" + "=" * 60)
print("  EXPERIMENT 4 COMPLETED SUCCESSFULLY")
print("=" * 60)
