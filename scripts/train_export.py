#!/usr/bin/env python3
"""Train 3-class LR on all labeled data, emit the model as SQL (weights table) + sklearn preds for validation."""
import numpy as np, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

t = pq.read_table('hn_sent_id.parquet')
ids = np.array(t.column('id').to_pylist())
y   = np.array(t.column('y').to_pylist())
X   = np.asarray(t.column('x').to_pylist(), dtype=np.float64)   # float64 to match CH scoring
Xn  = normalize(X)                                              # L2 per row

lr = LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced').fit(Xn, y)
classes = list(lr.classes_); coef = lr.coef_; intercept = lr.intercept_
print("classes:", classes, "| coef", coef.shape, "| train acc:", round((lr.predict(Xn)==y).mean(),4))

with open('sent_lr_model.sql','w') as f:
    f.write("DROP TABLE IF EXISTS default.sent_lr_model;\n")
    f.write("CREATE TABLE default.sent_lr_model (class String, w Array(Float64), b Float64) ENGINE=MergeTree ORDER BY class;\n")
    for k,c in enumerate(classes):
        w = "[" + ",".join(repr(float(v)) for v in coef[k]) + "]"
        f.write(f"INSERT INTO default.sent_lr_model VALUES ('{c}', {w}, {repr(float(intercept[k]))});\n")

with open('sk_pred.tsv','w') as f:
    for i,p in zip(ids, lr.predict(Xn)): f.write(f"{i}\t{p}\n")
print("wrote sent_lr_model.sql, sk_pred.tsv")
