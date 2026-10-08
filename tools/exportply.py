import pickle, numpy as np, sys

exp = sys.argv[1]
keep_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 97.0

d = pickle.load(open(f'experiments/{exp}/points.pickle','rb'))
P = np.loadtxt(f'experiments/{exp}/poses_pred.txt')[:, 1:4]
step = np.linalg.norm(np.diff(P, axis=0), axis=1)
thr_j = 5 * np.median(step)
keep = np.ones(len(P), bool)
keep[1:-1] = ~((step[:-1] > thr_j) & (step[1:] > thr_j))
print(f'dropped {(~keep).sum()} outlier pose(s)')
P = P[keep]

xyz = np.array([v[0] for v in d.values()], dtype=float)
rgb = np.clip(np.array([v[1] for v in d.values()], dtype=float) * 255, 0, 255).astype(np.uint8)

ok = np.isfinite(xyz).all(1)
xyz, rgb = xyz[ok], rgb[ok]

# Outlier cut by distance to the camera path, not the cloud centroid
dist = np.linalg.norm(xyz[:, None, :] - P[None, :, :], axis=2).min(1)
thr = np.percentile(dist, keep_pct)
m = dist <= thr
xyz, rgb = xyz[m], rgb[m]
print(f'{exp}: kept {len(xyz)}, cut {(~m).sum()} beyond {thr:.2f}')

out = f'experiments/{exp}/{exp}.ply'
with open(out, 'w') as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(xyz) + len(P)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")
    for p, c in zip(xyz, rgb):
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
    for p in P:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 255 0 0\n")
print('wrote', out)
