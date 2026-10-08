import numpy as np, pickle, sys, os

exp = sys.argv[1]

P = np.loadtxt(os.path.join(exp, 'poses_pred.txt'))
t = P[:, 1:4]
path = np.linalg.norm(np.diff(t, axis=0), axis=1).sum()
span = t.max(0) - t.min(0)
extent = np.linalg.norm(span)
print(f'--- {exp}')
print(f'keyframes {len(t)}')
print(f'path {path:.3f}  extent {extent:.3f}  ratio {path/extent:.2f}')
step = np.linalg.norm(np.diff(t, axis=0), axis=1)
nj = int((step > 5*np.median(step)).sum())
print(f'jumps >5x median: {nj}/{len(step)}' + ('   <-- TRAJECTORY SUSPECT' if nj > len(step)*0.05 else ''))
print(f'per-axis span  x {span[0]:.3f}  y {span[1]:.3f}  z {span[2]:.3f}')

with open(os.path.join(exp, 'points.pickle'), 'rb') as f:
    raw = pickle.load(f)
print('points.pickle type:', type(raw))

# Coerce whatever shape it is into an (N,3) array
if isinstance(raw, dict):
    vals = list(raw.values())
    print('  dict, n =', len(vals), '| sample:', repr(vals[0])[:120])
    try:
        pts = np.array([np.asarray(v).ravel()[:3] for v in vals], dtype=float)
    except Exception as e:
        print('  could not coerce:', e); sys.exit()
else:
    pts = np.asarray(raw, dtype=float)
    print('  array shape', pts.shape)
    if pts.ndim == 2 and pts.shape[1] > 3:
        pts = pts[:, :3]

pts = pts[np.isfinite(pts).all(1)]
print(f'points {len(pts)}')

c = np.median(pts, axis=0)
r = np.linalg.norm(pts - c, axis=1)
p10, p50, p90 = np.percentile(r, [10, 50, 90])
print(f'radial  p10 {p10:.3f}  median {p50:.3f}  p90 {p90:.3f}  SPREAD {p90/max(p10,1e-9):.2f}')

# Depth relative to the camera path, which is the thing that matters:
# a shell centered on the trajectory means no triangulation happened.
d = np.linalg.norm(pts[:, None, :] - t[None, :, :], axis=2).min(1)
q10, q50, q90 = np.percentile(d, [10, 50, 90])
print(f'dist-to-traj  p10 {q10:.3f}  median {q50:.3f}  p90 {q90:.3f}  SPREAD {q90/max(q10,1e-9):.2f}')
