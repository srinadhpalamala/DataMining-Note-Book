# ============================================================
# Experiment 1: KDD Process - Knowledge Discovery in Databases
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 1: KDD PROCESS DEMONSTRATION")
print("=" * 60)

# -------------------------------------------------------
# STEP 1: DATA SELECTION
# -------------------------------------------------------
print("\n[STEP 1] DATA SELECTION")
print("-" * 40)

# Sample dataset (Weather dataset)
data = {
    'Outlook':     ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast',
                    'Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                    'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':    ['High','High','High','High','Normal','Normal','Normal',
                    'High','Normal','Normal','Normal','High','Normal','High'],
    'Windy':       [False,True,False,False,False,True,True,
                    False,False,False,True,True,False,True],
    'Play':        ['No','No','Yes','Yes','Yes','No','Yes',
                    'No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\nShape: {df.shape[0]} instances, {df.shape[1]} attributes")

# -------------------------------------------------------
# STEP 2: DATA PREPROCESSING (DATA CLEANING)
# -------------------------------------------------------
print("\n[STEP 2] DATA PREPROCESSING")
print("-" * 40)

print(f"Missing Values:\n{df.isnull().sum()}")
print(f"\nDuplicate Records: {df.duplicated().sum()}")
df_clean = df.drop_duplicates()
print(f"Records after removing duplicates: {len(df_clean)}")

# -------------------------------------------------------
# STEP 3: DATA TRANSFORMATION
# -------------------------------------------------------
print("\n[STEP 3] DATA TRANSFORMATION")
print("-" * 40)

le = LabelEncoder()
df_encoded = df_clean.copy()
for col in ['Outlook', 'Temperature', 'Humidity', 'Play']:
    df_encoded[col] = le.fit_transform(df_encoded[col])
df_encoded['Windy'] = df_encoded['Windy'].astype(int)

print("Encoded Dataset:")
print(df_encoded)

# -------------------------------------------------------
# STEP 4: DATA MINING (Classification)
# -------------------------------------------------------
print("\n[STEP 4] DATA MINING - Decision Tree Classification")
print("-" * 40)

X = df_encoded.drop('Play', axis=1)
y = df_encoded['Play']

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(f"Decision Tree Accuracy: {acc * 100:.2f}%")
print(f"Feature Importances:")
for feat, imp in zip(X.columns, clf.feature_importances_):
    print(f"   {feat}: {imp:.4f}")

# -------------------------------------------------------
# STEP 5: PATTERN EVALUATION
# -------------------------------------------------------
print("\n[STEP 5] PATTERN EVALUATION")
print("-" * 40)

important_features = [f for f, i in zip(X.columns, clf.feature_importances_) if i > 0.1]
print(f"Most Important Features (importance > 0.1): {important_features}")

# -------------------------------------------------------
# STEP 6: KNOWLEDGE PRESENTATION
# -------------------------------------------------------
print("\n[STEP 6] KNOWLEDGE PRESENTATION")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Feature Importance
axes[0].bar(X.columns, clf.feature_importances_, color='steelblue', edgecolor='black')
axes[0].set_title('Feature Importances (KDD - Data Mining Step)')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Importance')
axes[0].tick_params(axis='x', rotation=15)

# Class Distribution
play_counts = df['Play'].value_counts()
axes[1].pie(play_counts, labels=play_counts.index, autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff'], startangle=90)
axes[1].set_title('Class Distribution (Play)')

plt.tight_layout()
plt.savefig('exp1_kdd_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'exp1_kdd_output.png'")

print("\n" + "=" * 60)
print("  KDD PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)
