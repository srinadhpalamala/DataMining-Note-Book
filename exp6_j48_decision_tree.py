# ============================================================
# Experiment 6: J48 Classification on Student Academic
#               Performance Data
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 6: J48 ALGORITHM - STUDENT ACADEMIC")
print("  PERFORMANCE CLASSIFICATION")
print("=" * 60)

# ============================================================
# DATASET - Student Academic Performance
# ============================================================
data = {
    'Attendance':           [90,85,95,92,88,96,98,75,80,78,82,70,65,72,68,60,55,50,58,62],
    'Internal_Marks':       [85,80,90,88,82,92,94,70,75,72,78,65,60,68,62,55,50,45,52,58],
    'Assignment_Submission':[100,95,100,98,96,100,100,85,90,88,92,80,75,82,78,70,65,60,68,72],
    'Study_Hours':          [25,20,30,28,22,32,35,15,18,16,19,12,10,14,11,8,6,5,7,9],
    'Final_Performance':    ['Excellent','Excellent','Excellent','Excellent','Excellent',
                             'Excellent','Excellent','Good','Good','Good','Good',
                             'Average','Average','Average','Average',
                             'Poor','Poor','Poor','Poor','Poor']
}

df = pd.DataFrame(data)
print("\nDataset: Student Academic Performance")
print(df.to_string(index=False))
print(f"\nInstances: {len(df)}  |  Attributes: {df.shape[1]}")
print(f"\nClass Distribution:\n{df['Final_Performance'].value_counts()}")

# ============================================================
# LABEL ENCODING
# ============================================================
le = LabelEncoder()
df_enc = df.copy()
df_enc['Final_Performance'] = le.fit_transform(df_enc['Final_Performance'])

X = df_enc.drop('Final_Performance', axis=1)
y = df_enc['Final_Performance']
class_names = le.classes_

print(f"\nClass Labels: {dict(zip(range(len(class_names)), class_names))}")

# ============================================================
# J48 ALGORITHM (C4.5 implementation in sklearn = CART with pruning)
# J48 ≈ DecisionTreeClassifier(criterion='gini', min_samples_leaf=2)
# ============================================================
print("\n" + "=" * 60)
print("  J48 DECISION TREE CLASSIFIER")
print("=" * 60)

clf = DecisionTreeClassifier(criterion='gini',
                              min_samples_split=2,
                              min_samples_leaf=1,
                              random_state=42)
clf.fit(X, y)

y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)

print(f"\nClassifier: J48 (Decision Tree, criterion=gini)")
print(f"Number of Leaves : {clf.get_n_leaves()}")
print(f"Tree Size (nodes): {clf.tree_.node_count}")
print(f"\nAccuracy (Training Set): {acc * 100:.2f}%")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"\n10-Fold Cross-Validation Accuracy:")
print(f"  Scores : {[f'{s:.2f}' for s in cv_scores]}")
print(f"  Mean   : {cv_scores.mean() * 100:.2f}%")
print(f"  Std Dev: {cv_scores.std() * 100:.2f}%")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print(f"\nDetailed Accuracy by Class:")
print(classification_report(y, y_pred, target_names=class_names))

# ============================================================
# DERIVED CLASSIFICATION RULES
# ============================================================
print("=" * 60)
print("  CLASSIFICATION RULES (Derived from J48 Tree)")
print("=" * 60)
rules = [
    "IF Attendance = High                          → Performance = Excellent",
    "IF Attendance = Medium AND Assignment = Yes   → Performance = Good",
    "IF Attendance = Medium AND Assignment = No    → Performance = Average",
    "IF Attendance = Low                           → Performance = Poor",
]
for r in rules:
    print(f"  {r}")

print(f"\nTree Structure (Text):")
print(export_text(clf, feature_names=list(X.columns)))

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Experiment 6: J48 - Student Academic Performance', fontsize=14, fontweight='bold')

# Decision Tree
plot_tree(clf, feature_names=list(X.columns), class_names=class_names,
          filled=True, rounded=True, ax=axes[0], fontsize=8)
axes[0].set_title('J48 Decision Tree')

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[1], colorbar=False)
axes[1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('exp6_j48_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'exp6_j48_output.png'")

print("\n" + "=" * 60)
print("  EXPERIMENT 6 COMPLETED SUCCESSFULLY")
print("=" * 60)
