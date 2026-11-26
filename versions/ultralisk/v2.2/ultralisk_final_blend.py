from pathlib import Path
import pandas as pd
import numpy as np
import hashlib

# Place both original team submissions in the working directory.
ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path(__file__).resolve().parent
A = ROOT / 'submission_v3_t4x2.csv'
B = ROOT / 'submission_optimized.csv'
if not B.exists():
    B = ROOT / 'Ultralisk-v2.1' / B.name
OUT = ROOT / 'submission_ultralisk_v2_2_final.csv'
cols = ['RH', 'T', 'U', 'V']
a, b = pd.read_csv(A), pd.read_csv(B)
for df in (a,b):
    assert df.shape == (646800,5)
    assert df.columns.tolist() == ['id', *cols]
    assert df.id.is_unique and np.isfinite(df[cols].to_numpy()).all()
assert a.id.equals(b.id)
# Fixed conservative weight, not fitted on hidden labels.
x = 0.60*a[cols].to_numpy() + 0.40*b[cols].to_numpy()
x[:,0] = np.clip(x[:,0],0,100)
result = pd.DataFrame(x, columns=cols)
result.insert(0,'id',a.id)
result.to_csv(OUT,index=False,float_format='%.5f')
check = pd.read_csv(OUT)
assert check.id.equals(a.id) and check.shape == (646800,5)
assert np.isfinite(check[cols].to_numpy()).all() and check.RH.between(0,100).all()
print(OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())
