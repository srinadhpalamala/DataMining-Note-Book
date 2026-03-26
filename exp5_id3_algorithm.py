# ============================================================
# Experiment 5: Classification using ID3 Algorithm
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 5: ID3 CLASSIFICATION ALGORITHM")
print("=" * 60)

# ============================================================
# DATASET - Weather Nominal (Same as WEKA weather.nominal.arff)
# ============================================================
data = {
    'Outlook':     ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast',
                    'Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                    'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':    ['High','High','High','High','Normal','Normal','Normal',
                    'High','Normal','Normal','Normal','High','Normal','High'],
    'Windy':       ['FALSE','TRUE','FALSE','FALSE','FALSE','TRUE','TRUE',
                    'FALSE','FALSE','FALSE','TRUE','TRUE','FALSE','TRUE'],
    'Play':        ['No','No','Yes','Yes','Yes','No','Yes',
                    'No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)
print("\nDataset: Weather Nominal")
print(df.to_string(index=False))
print(f"\nInstances: {len(df)}  |  Attributes: {df.shape[1]}")

# ============================================================
# ENTROPY & INFORMATION GAIN (ID3 Core Logic)
# ============================================================
print("\n" + "=" * 60)
print("  ID3 ALGORITHM - ENTROPY & INFORMATION GAIN")
print("=" * 60)

def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def information_gain(df, feature, target='Play'):
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for val in values:
        subset = df[df[feature] == val]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy

total_ent = entropy(df['Play'])
print(f"\nTotal Entropy of 'Play': {total_ent:.4f}")
print("\nInformation Gain for each attribute:")
features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
ig_values = {}
for feat in features:
    ig = information_gain(df, feat)
    ig_values[feat] = ig
    print(f"  {feat:15s}: {ig:.4f}")

best_feature = max(ig_values, key=ig_values.get)
print(f"\n>>> Root Node (Best Split): '{best_feature}' (IG = {ig_values[best_feature]:.4f})")

# ============================================================
# CLASSIFICATION RULES (Manually Derived from Tree)
# ============================================================
print("\n" + "=" * 60)
print("  CLASSIFICATION RULES (from ID3 Decision Tree)")
print("=" * 60)
rules = [
    "IF Outlook = Sunny  AND Humidity = High   → Play = No",
    "IF Outlook = Sunny  AND Humidity = Normal → Play = Yes",
    "IF Outlook = Overcast                     → Play = Yes",
    "IF Outlook = Rainy  AND Windy = TRUE      → Play = No",
    "IF Outlook = Rainy  AND Windy = FALSE     → Play = Yes",
]
for rule in rules:
    print(f"  {rule}")

# ============================================================
# SKLEARN IMPLEMENTATION (ID3 = criterion='entropy')
# ============================================================
print("\n" + "=" * 60)
print("  SKLEARN DECISION TREE (criterion=entropy = ID3)")
print("=" * 60)

le = LabelEncoder()
df_enc = df.copy()
for col in df_enc.columns:
    df_enc[col] = le.fit_transform(df_enc[col])

X = df_enc.drop('Play', axis=1)
y = df_enc['Play']

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

y_pred = clf.predict(X)
print(f"\nAccuracy (on training set): {accuracy_score(y, y_pred) * 100:.2f}%")
print(f"\nDecision Tree Rules (Text):")
print(export_text(clf, feature_names=list(X.columns)))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
print(f"Confusion Matrix:\n{cm}")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Experiment 5: ID3 Classification Algorithm', fontsize=14, fontweight='bold')

# Decision Tree Plot
plot_tree(clf, feature_names=list(X.columns), class_names=['No','Yes'],
          filled=True, ax=axes[0])
axes[0].set_title('ID3 Decision Tree')

# Information Gain Bar Chart
axes[1].barh(list(ig_values.keys()), list(ig_values.values()), color='steelblue', edgecolor='black')
axes[1].set_title('Information Gain per Attribute')
axes[1].set_xlabel('Information Gain')
axes[1].axvline(x=0, color='black', linewidth=0.5)
for i, (feat, val) in enumerate(ig_values.items()):
    axes[1].text(val + 0.002, i, f'{val:.4f}', va='center')

plt.tight_layout()
plt.savefig('exp5_id3_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'exp5_id3_output.png'")

print("\n" + "=" * 60)
print("  EXPERIMENT 5 COMPLETED SUCCESSFULLY")
print("=" * 60)
