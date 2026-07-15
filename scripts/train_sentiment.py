#!/usr/bin/env python3
"""Train sentiment classifier on qwen3 embeddings (labels from aiClassify).
Compares: majority-class, cosine-to-prototype (nearest centroid), logistic regression.
Input: parquet with columns y (label String) and x (Array(Float32) embedding)."""
import sys, numpy as np
import pyarrow.parquet as pq
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

path = sys.argv[1]
t = pq.read_table(path)
y = np.array(t.column('y').to_pylist())
X = np.asarray(t.column('x').to_pylist(), dtype=np.float32)
print(f"loaded X={X.shape}  classes={dict(sorted(Counter(y).items()))}")

# raw embedding norm (are they unit-norm already?)
n = np.linalg.norm(X, axis=1)
print(f"embedding L2 norm: mean {n.mean():.3f} min {n.min():.3f} max {n.max():.3f}")

Xn = normalize(X)  # L2-normalize rows (cosine geometry)
Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.25, random_state=42, stratify=y)
print(f"train {Xtr.shape[0]}  test {Xte.shape[0]}")
classes = np.array(sorted(np.unique(y)))

# 1) majority class
maj = Counter(ytr).most_common(1)[0][0]
acc_maj = accuracy_score(yte, np.full(len(yte), maj))

# 2) cosine-to-prototype (class-mean centroid, cosine) — the "topic-dominated" baseline
cents = normalize(np.stack([Xtr[ytr == c].mean(0) for c in classes]))
pc = classes[(Xte @ cents.T).argmax(1)]

# 3) logistic regression (learned linear head)
lr = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')
lr.fit(Xtr, ytr)
pl = lr.predict(Xte)

print("\n================ RESULTS (test set) ================")
print(f"{'majority-class':<26} acc {acc_maj:.3f}")
print(f"{'cosine-to-prototype':<26} acc {accuracy_score(yte,pc):.3f}   macro-F1 {f1_score(yte,pc,average='macro'):.3f}")
print(f"{'logistic regression':<26} acc {accuracy_score(yte,pl):.3f}   macro-F1 {f1_score(yte,pl,average='macro'):.3f}")
print("\n--- logistic regression, per-class ---")
print(classification_report(yte, pl, digits=3))
print(f"confusion matrix rows=true {list(classes)}:")
print(confusion_matrix(yte, pl, labels=classes))
