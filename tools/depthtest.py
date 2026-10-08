import pickle, numpy as np, sys
from scipy.spatial.transform import Rotation as R

exp = sys.argv[1]
pts = pickle.load(open(f'experiments/{exp}/points.pickle','rb'))
ppm = pickle.load(open(f'experiments/{exp}/pose_point_map.pickle','rb'))
pos = pickle.load(open(f'experiments/{exp}/poses.pickle','rb'))

def to_Rt(v):
    v = np.asarray(v[0] if isinstance(v, tuple) else v, dtype=float).ravel()
    if v.size == 7:                      # tx ty tz qx qy qz qw
        return R.from_quat(v[3:7]).as_matrix(), v[:3]
    if v.size == 16:
        T = v.reshape(4,4); return T[:3,:3], T[:3,3]
    if v.size == 12:
        T = v.reshape(3,4); return T[:3,:3], T[:3,3]
    raise ValueError(f'unexpected pose size {v.size}')

# first observing frame per point
first = {}
for f in sorted(ppm.keys()):
    if f not in pos: continue
    for pid, _ in ppm[f]:
        first.setdefault(pid, f)

d = []
for pid, f in first.items():
    if pid not in pts: continue
    X = np.asarray(pts[pid][0], dtype=float)
    Rm, t = to_Rt(pos[f])
    d.append((Rm.T @ (X - t))[2])        # camera-frame z; flip to (Rm@X + t) if convention differs
d = np.array([x for x in d if np.isfinite(x)])

print(f'--- {exp}   n={len(d)}')
print(f'cam-frame depth  p10 {np.percentile(d,10):.3f}  median {np.median(d):.3f}  p90 {np.percentile(d,90):.3f}')
print(f'  SPREAD p90/p10 {np.percentile(d,90)/max(np.percentile(d,10),1e-9):.2f}')
print(f'  within 5% of seed 10.0: {100*np.mean(np.abs(d-10)<0.5):.1f}%')
print(f'  within 20% of seed:     {100*np.mean(np.abs(d-10)<2.0):.1f}%')
