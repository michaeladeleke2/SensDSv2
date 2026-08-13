import numpy as np, glob, os
p = os.path.expanduser("~/SensDSv2_data")
raws = sorted(glob.glob(f"{p}/**/*_raw.npy", recursive=True))
print(f"{len(raws)} raw cubes found\n")
shapes = set()
for f in raws:
    a = np.load(f)
    shapes.add((a.shape, str(a.dtype)))
    print(f"{os.path.basename(f):30s} {a.shape}  {a.dtype}  {a.nbytes/1e6:.1f} MB")
print(f"\nUnique shape/dtype combos: {len(shapes)}")
for s in shapes:
    print(" ", s)