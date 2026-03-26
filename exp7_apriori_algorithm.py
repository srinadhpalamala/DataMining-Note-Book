# ============================================================
# Experiment 7: Association Rule Mining using Apriori Algorithm
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 7: APRIORI ALGORITHM")
print("  ASSOCIATION RULE MINING - CREDIT CARD / SUPERMARKET DATA")
print("=" * 60)

# ============================================================
# DATASET - Transaction Data (Simulating Supermarket/Credit Card)
# ============================================================
transactions = [
    ['Bread', 'Butter', 'Milk'],
    ['Bread', 'Butter'],
    ['Bread', 'Milk', 'Eggs'],
    ['Butter', 'Milk', 'Eggs'],
    ['Bread', 'Butter', 'Milk', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter', 'Eggs'],
    ['Butter', 'Eggs'],
    ['Bread', 'Milk'],
    ['Bread', 'Butter', 'Milk'],
    ['Eggs', 'Milk', 'Butter'],
    ['Bread', 'Butter'],
    ['Bread', 'Milk', 'Butter', 'Eggs'],
    ['Milk', 'Butter'],
]

print(f"\nTransaction Dataset: {len(transactions)} transactions")
print("Items: Bread, Butter, Milk, Eggs")
for i, t in enumerate(transactions, 1):
    print(f"  T{i:02d}: {t}")

# ============================================================
# APRIORI IMPLEMENTATION FROM SCRATCH
# ============================================================
MIN_SUPPORT    = 0.3   # 30%
MIN_CONFIDENCE = 0.6   # 60%
n_transactions = len(transactions)

print(f"\nParameters:")
print(f"  Minimum Support    : {MIN_SUPPORT * 100:.0f}%")
print(f"  Minimum Confidence : {MIN_CONFIDENCE * 100:.0f}%")

def get_support(itemset, transactions):
    count = sum(1 for t in transactions if set(itemset).issubset(set(t)))
    return count / len(transactions)

def apriori(transactions, min_support):
    # Get all unique items
    all_items = sorted(set(item for t in transactions for item in t))
    # C1 - Candidate 1-itemsets
    freq_itemsets = {}
    L1 = []
    for item in all_items:
        sup = get_support([item], transactions)
        if sup >= min_support:
            L1.append(frozenset([item]))
            freq_itemsets[frozenset([item])] = sup

    current_Lk = L1
    k = 2
    while current_Lk:
        # Generate Ck
        Ck = []
        current_list = list(current_Lk)
        for i in range(len(current_list)):
            for j in range(i + 1, len(current_list)):
                union = current_list[i] | current_list[j]
                if len(union) == k:
                    Ck.append(union)

        # Filter by min support
        Lk = []
        for itemset in Ck:
            sup = get_support(list(itemset), transactions)
            if sup >= min_support:
                Lk.append(itemset)
                freq_itemsets[itemset] = sup

        current_Lk = Lk
        k += 1

    return freq_itemsets

def generate_rules(freq_itemsets, min_confidence):
    rules = []
    for itemset in freq_itemsets:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if consequent:
                    conf = freq_itemsets[itemset] / freq_itemsets.get(antecedent, 0)
                    if conf >= min_confidence:
                        lift = conf / freq_itemsets.get(consequent, 1)
                        rules.append({
                            'Antecedent': set(antecedent),
                            'Consequent': set(consequent),
                            'Support':    round(freq_itemsets[itemset], 4),
                            'Confidence': round(conf, 4),
                            'Lift':       round(lift, 4)
                        })
    return rules

# ============================================================
# RUN APRIORI
# ============================================================
print("\n" + "=" * 60)
print("  STEP 1: FREQUENT ITEMSETS")
print("=" * 60)

freq_itemsets = apriori(transactions, MIN_SUPPORT)

# Print by size
for k in range(1, 5):
    k_itemsets = {k_: v for k_, v in freq_itemsets.items() if len(k_) == k}
    if k_itemsets:
        print(f"\n  L{k} - Frequent {k}-Itemsets:")
        print(f"  {'Itemset':<35} {'Support':>10}  {'Count':>6}")
        print("  " + "-" * 55)
        for itemset, sup in sorted(k_itemsets.items(), key=lambda x: -x[1]):
            count = int(sup * n_transactions)
            print(f"  {str(set(itemset)):<35} {sup:>10.4f}  {count:>6}")

# ============================================================
# ASSOCIATION RULES
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: ASSOCIATION RULES GENERATED")
print("=" * 60)

rules = generate_rules(freq_itemsets, MIN_CONFIDENCE)
rules_df = pd.DataFrame(rules)
rules_df = rules_df.sort_values('Confidence', ascending=False).reset_index(drop=True)

print(f"\nTotal Rules Generated: {len(rules)}")
print(f"\n{'#':<4} {'Antecedent':<20} {'Consequent':<15} {'Support':>8} {'Confidence':>11} {'Lift':>6}")
print("-" * 70)
for i, row in rules_df.iterrows():
    ant = str(row['Antecedent']).replace("'","").replace("{","").replace("}","")
    con = str(row['Consequent']).replace("'","").replace("{","").replace("}","")
    print(f"{i+1:<4} {ant:<20} {con:<15} {row['Support']:>8.4f} {row['Confidence']:>11.4f} {row['Lift']:>6.4f}")

# ============================================================
# INTERPRETATION
# ============================================================
print("\n" + "=" * 60)
print("  INTERPRETATION")
print("=" * 60)
print("""
  - Support    : How frequently the items appear together
  - Confidence : If customer buys Antecedent, probability of buying Consequent
  - Lift > 1   : Positive association (items are related)
  - Lift = 1   : No association
  - Lift < 1   : Negative association

  Business Insights:
  → Place frequently associated items near each other in store
  → Use rules for cross-selling and combo offers
  → Design targeted promotions based on purchasing patterns
""")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Experiment 7: Apriori - Association Rule Mining', fontsize=14, fontweight='bold')

# Support of frequent itemsets
labels = [str(set(k)).replace("'","") for k in freq_itemsets.keys() if len(k) >= 2]
values = [v for k, v in freq_itemsets.items() if len(k) >= 2]
axes[0].barh(labels, values, color='steelblue', edgecolor='black')
axes[0].axvline(x=MIN_SUPPORT, color='red', linestyle='--', label=f'Min Support={MIN_SUPPORT}')
axes[0].set_title('Frequent Itemsets - Support Values')
axes[0].set_xlabel('Support')
axes[0].legend()

# Confidence scatter plot
if len(rules_df) > 0:
    axes[1].scatter(rules_df['Support'], rules_df['Confidence'],
                    c=rules_df['Lift'], cmap='YlOrRd', s=100, edgecolors='black')
    axes[1].axhline(y=MIN_CONFIDENCE, color='red', linestyle='--',
                    label=f'Min Confidence={MIN_CONFIDENCE}')
    axes[1].set_title('Support vs Confidence (color=Lift)')
    axes[1].set_xlabel('Support')
    axes[1].set_ylabel('Confidence')
    axes[1].legend()
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Lift')

plt.tight_layout()
plt.savefig('exp7_apriori_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved as 'exp7_apriori_output.png'")

print("\n" + "=" * 60)
print("  EXPERIMENT 7 COMPLETED SUCCESSFULLY")
print("=" * 60)
