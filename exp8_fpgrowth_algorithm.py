# ============================================================
# Experiment 8: Association Rules using FP-Growth Algorithm
#               on Contact Lenses Dataset
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 8: FP-GROWTH ALGORITHM")
print("  CONTACT LENSES DATASET")
print("=" * 60)

# ============================================================
# CONTACT LENSES DATASET
# ============================================================
data = {
    'age_young':          ['yes','yes','yes','yes','yes','yes','yes','yes',
                           'no','no','no','no','no','no','no','no',
                           'no','no','no','no','no','no','no','no'],
    'age_pre_presby':     ['no','no','no','no','no','no','no','no',
                           'yes','yes','yes','yes','yes','yes','yes','yes',
                           'no','no','no','no','no','no','no','no'],
    'age_presby':         ['no','no','no','no','no','no','no','no',
                           'no','no','no','no','no','no','no','no',
                           'yes','yes','yes','yes','yes','yes','yes','yes'],
    'spec_myope':         ['yes','yes','yes','yes','no','no','no','no',
                           'yes','yes','yes','yes','no','no','no','no',
                           'yes','yes','yes','yes','no','no','no','no'],
    'spec_hypermetrope':  ['no','no','no','no','yes','yes','yes','yes',
                           'no','no','no','no','yes','yes','yes','yes',
                           'no','no','no','no','yes','yes','yes','yes'],
    'astigmatism_no':     ['yes','yes','no','no','yes','yes','no','no',
                           'yes','yes','no','no','yes','yes','no','no',
                           'yes','yes','no','no','yes','yes','no','no'],
    'astigmatism_yes':    ['no','no','yes','yes','no','no','yes','yes',
                           'no','no','yes','yes','no','no','yes','yes',
                           'no','no','yes','yes','no','no','yes','yes'],
    'tear_reduced':       ['yes','no','yes','no','yes','no','yes','no',
                           'yes','no','yes','no','yes','no','yes','no',
                           'yes','no','yes','no','yes','no','yes','no'],
    'tear_normal':        ['no','yes','no','yes','no','yes','no','yes',
                           'no','yes','no','yes','no','yes','no','yes',
                           'no','yes','no','yes','no','yes','no','yes'],
    'lens_soft':          ['no','yes','no','no','no','yes','no','no',
                           'no','yes','no','no','no','yes','no','no',
                           'no','yes','no','no','no','yes','no','no'],
    'lens_hard':          ['no','no','yes','no','no','no','yes','no',
                           'no','no','yes','no','no','no','yes','no',
                           'no','no','no','no','no','no','no','no'],
    'lens_none':          ['yes','no','no','yes','yes','no','no','yes',
                           'yes','no','no','yes','yes','no','no','yes',
                           'yes','no','yes','yes','yes','no','yes','yes'],
}

df = pd.DataFrame(data)
print(f"\nDataset: Contact Lenses")
print(f"Instances: {len(df)}  |  Attributes: {df.shape[1]}")
print(f"\nFirst 8 rows:")
print(df.head(8).to_string())

# ============================================================
# FP-TREE NODE CLASS
# ============================================================
class FPTreeNode:
    def __init__(self, item, count, parent):
        self.item   = item
        self.count  = count
        self.parent = parent
        self.children = {}
        self.link   = None

class FPTree:
    def __init__(self, min_support):
        self.min_support = min_support
        self.root = FPTreeNode(None, 0, None)
        self.header_table = defaultdict(list)
        self.freq_items = {}

    def build(self, transactions):
        # Count item frequencies
        item_count = defaultdict(int)
        for t in transactions:
            for item in t:
                item_count[item] += 1

        # Filter by min support
        n = len(transactions)
        self.freq_items = {k: v for k, v in item_count.items()
                           if v / n >= self.min_support}
        if not self.freq_items:
            return

        # Insert transactions into tree
        for t in transactions:
            filtered = sorted([item for item in t if item in self.freq_items],
                               key=lambda x: self.freq_items[x], reverse=True)
            if filtered:
                self._insert(filtered, self.root)

    def _insert(self, items, node):
        if not items:
            return
        first = items[0]
        if first in node.children:
            node.children[first].count += 1
        else:
            new_node = FPTreeNode(first, 1, node)
            node.children[first] = new_node
            self.header_table[first].append(new_node)
        self._insert(items[1:], node.children[first])

    def mine_patterns(self, suffix, min_sup_count, n):
        patterns = {}
        for item in self.freq_items:
            new_pattern = suffix + [item]
            sup = self.freq_items[item] / n
            if sup >= self.min_support:
                patterns[frozenset(new_pattern)] = round(sup, 4)
        return patterns

# ============================================================
# Convert dataset to transactions (only 'yes' attributes)
# ============================================================
transactions = []
for _, row in df.iterrows():
    trans = [col for col in df.columns if row[col] == 'yes']
    transactions.append(trans)

MIN_SUPPORT    = 0.4
MIN_CONFIDENCE = 0.6
n = len(transactions)

print(f"\nParameters:")
print(f"  Min Support    : {MIN_SUPPORT * 100:.0f}%")
print(f"  Min Confidence : {MIN_CONFIDENCE * 100:.0f}%")

# ============================================================
# BUILD FP-TREE & MINE FREQUENT ITEMSETS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 1: FREQUENT ITEMSETS (FP-Tree Mining)")
print("=" * 60)

fp = FPTree(MIN_SUPPORT)
fp.build(transactions)

# Mine single item frequent itemsets
freq_1 = {frozenset([k]): round(v / n, 4) for k, v in fp.freq_items.items()}
print(f"\n  1-Itemsets (support >= {MIN_SUPPORT}):")
for itemset, sup in sorted(freq_1.items(), key=lambda x: -x[1]):
    print(f"    {set(itemset)} : {sup:.4f} (count={int(sup*n)})")

# Mine 2-itemsets
print(f"\n  2-Itemsets:")
freq_2 = {}
keys = list(fp.freq_items.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        pair = frozenset([keys[i], keys[j]])
        count = sum(1 for t in transactions if set(pair).issubset(set(t)))
        sup = count / n
        if sup >= MIN_SUPPORT:
            freq_2[pair] = round(sup, 4)
            print(f"    {set(pair)} : {sup:.4f} (count={count})")

all_freq = {**freq_1, **freq_2}

# ============================================================
# GENERATE ASSOCIATION RULES
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: ASSOCIATION RULES GENERATED")
print("=" * 60)

rules = []
for itemset in all_freq:
    if len(itemset) < 2:
        continue
    items = list(itemset)
    for i in range(1, len(items)):
        from itertools import combinations as combs
        for ant in combs(items, i):
            ant = frozenset(ant)
            con = itemset - ant
            if con and ant in all_freq:
                conf = all_freq[itemset] / all_freq[ant]
                lift = conf / all_freq.get(con, 1)
                conv = (1 - all_freq.get(con, 1)) / (1 - conf + 1e-9)
                if conf >= MIN_CONFIDENCE:
                    rules.append({
                        'Antecedent': set(ant),
                        'Consequent': set(con),
                        'Support':    round(all_freq[itemset], 4),
                        'Confidence': round(conf, 4),
                        'Lift':       round(lift, 4)
                    })

rules_df = pd.DataFrame(rules).sort_values('Confidence', ascending=False).reset_index(drop=True)
print(f"\nTotal Rules Found: {len(rules_df)}")
if len(rules_df) > 0:
    print(f"\n{'#':<4} {'Antecedent':<22} {'Consequent':<18} {'Sup':>6} {'Conf':>6} {'Lift':>6}")
    print("-" * 65)
    for i, row in rules_df.iterrows():
        a = str(row['Antecedent']).replace("'","")
        c = str(row['Consequent']).replace("'","")
        print(f"{i+1:<4} {a:<22} {c:<18} {row['Support']:>6.4f} {row['Confidence']:>6.4f} {row['Lift']:>6.4f}")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Experiment 8: FP-Growth - Contact Lenses Association Rules', fontsize=13, fontweight='bold')

# Frequent item support bar chart
items = list(fp.freq_items.keys())
supports = [fp.freq_items[i] / n for i in items]
colors = ['steelblue' if s >= MIN_SUPPORT else 'lightgray' for s in supports]
axes[0].bar(items, supports, color=colors, edgecolor='black')
axes[0].axhline(y=MIN_SUPPORT, color='red', linestyle='--', label=f'Min Support={MIN_SUPPORT}')
axes[0].set_title('Item Frequencies (Support)')
axes[0].set_ylabel('Support')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()

# Rule confidence bar chart
if len(rules_df) > 0:
    rule_labels = [f"R{i+1}" for i in range(len(rules_df))]
    axes[1].bar(rule_labels, rules_df['Confidence'], color='orange', edgecolor='black')
    axes[1].axhline(y=MIN_CONFIDENCE, color='red', linestyle='--',
                    label=f'Min Confidence={MIN_CONFIDENCE}')
    axes[1].set_title('Association Rules - Confidence')
    axes[1].set_ylabel('Confidence')
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, 'No rules generated\nat this threshold',
                 ha='center', va='center', transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig('exp8_fpgrowth_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'exp8_fpgrowth_output.png'")

print("\n" + "=" * 60)
print("  EXPERIMENT 8 COMPLETED SUCCESSFULLY")
print("=" * 60)
