#!/usr/bin/env python3
"""Topic classifier on qwen3 embeddings. Eval (majority/cosine/LR) + train final model -> topic_lr_model.parquet."""
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score, classification_report

t = pq.read_table('hn_topic.parquet')
y = np.array(t.column('y').to_pylist())
X = np.asarray(t.column('x').to_pylist(), dtype=np.float64)
Xn = normalize(X)
print("n=", len(y), "classes:", dict(sorted(Counter(y).items())))

Xtr,Xte,ytr,yte = train_test_split(Xn, y, test_size=0.25, random_state=42, stratify=y)
classes = np.array(sorted(np.unique(y)))
maj = Counter(ytr).most_common(1)[0][0]
cents = normalize(np.stack([Xtr[ytr==c].mean(0) for c in classes])); pc = classes[(Xte@cents.T).argmax(1)]
lr = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced').fit(Xtr, ytr); pl = lr.predict(Xte)
print(f"\nmajority-class        acc {accuracy_score(yte, np.full(len(yte),maj)):.3f}")
print(f"cosine-to-prototype   acc {accuracy_score(yte,pc):.3f}  macro-F1 {f1_score(yte,pc,average='macro'):.3f}")
print(f"logistic regression   acc {accuracy_score(yte,pl):.3f}  macro-F1 {f1_score(yte,pl,average='macro'):.3f}")
print("\n--- LR per-class ---"); print(classification_report(yte, pl, digits=3))

# final model on all data -> parquet
lrf = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced').fit(Xn, y)
cl = list(lrf.classes_)
pq.write_table(pa.table({
  'class': cl,
  'w': pa.array([[float(v) for v in lrf.coef_[k]] for k in range(len(cl))], type=pa.list_(pa.float64())),
  'b': [float(v) for v in lrf.intercept_]}), 'topic_lr_model.parquet')
print("wrote topic_lr_model.parquet; classes:", cl, "| full-data train acc:", round((lrf.predict(Xn)==y).mean(),4))
