import pickle, numpy as np, cv2, sys, os
from PIL import Image
from transformers import pipeline

exp, DS = sys.argv[1], os.environ['DS2']
STRIDE, GRID, VOX = 3, 2, 0.05   # keyframe stride, pixel stride, voxel size

poses = pickle.load(open(f'experiments/{exp}/poses.pickle','rb'))
pts   = pickle.load(open(f'experiments/{exp}/points.pickle','rb'))
ppm   = pickle.load(open(f'experiments/{exp}/pose_point_map.pickle','rb'))

# drop out-and-back spike poses
keys = sorted(poses.keys())
C = np.array([poses[k][0][:3,3] for k in keys])
s = np.linalg.norm(np.diff(C,axis=0),axis=1); m = np.median(s); j = s > 5*m
bad = set()
for i in range(1, len(keys)-1):
    if j[i-1] and j[i]: bad.add(keys[i])
print(f'excluding {len(bad)} spike poses of {len(keys)}')

dep = pipeline('depth-estimation',
               model='depth-anything/Depth-Anything-V2-Small-hf', device=0)

P_all, C_all, used, skipped = [], [], 0, 0
for n, k in enumerate(keys):
    if k in bad or n % STRIDE: continue
    T, K = poses[k]
    R, t = T[:3,:3], T[:3,3]
    fx, fy, cx, cy = K

    img = Image.open(f'{DS}/images/{k:08d}.jpg')
    W, H = img.size
    d_out = dep(img)['predicted_depth']
    pred = np.asarray(d_out.squeeze().float().cpu() if hasattr(d_out,'cpu') else d_out)
    if pred.shape != (H, W):
        pred = cv2.resize(pred.astype(np.float32), (W, H))

    obs = [(px, pts[pid][0]) for pid, px in ppm.get(k, []) if pid in pts]
    if len(obs) < 30: skipped += 1; continue
    uv = np.array([o[0] for o in obs], float)
    X  = np.array([o[1] for o in obs], float)
    z  = ((X - t) @ R)[:, 2]                       # camera-frame depth
    ok = (z > 0.2) & (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)
    if ok.sum() < 30: skipped += 1; continue

    p_s = pred[uv[ok,1].astype(int), uv[ok,0].astype(int)]
    invz = 1.0/z[ok]
    good_s = p_s > 1e-3
    if good_s.sum() < 30: skipped += 1; continue
    a = np.median(p_s[good_s] / invz[good_s])          # robust scale-only fit
    if not np.isfinite(a) or a <= 1e-6: skipped += 1; continue
    resid = np.abs(np.log(np.maximum(p_s[good_s],1e-6)) - np.log(a*invz[good_s]))
    if np.median(resid) > 0.35: skipped += 1; continue  # reject inconsistent frames

    inv = pred / a
    Z = np.where(inv > 1e-3, 1.0/np.maximum(inv, 1e-3), 0)
    lo, hi = np.percentile(z[ok], [2, 98])
    v, u = np.mgrid[0:H:GRID, 0:W:GRID]
    zz = Z[v, u]
    good = (zz > lo*0.5) & (zz < hi*1.5)
    u, v, zz = u[good], v[good], zz[good]
    if len(zz) == 0: skipped += 1; continue

    Xc = np.stack([(u-cx)/fx*zz, (v-cy)/fy*zz, zz], 1)
    P_all.append(Xc @ R.T + t)
    C_all.append(np.array(img)[v, u])
    used += 1

print(f'used {used} frames, skipped {skipped}')
P = np.vstack(P_all); Cc = np.vstack(C_all)
print(f'{len(P)} raw points')

q = np.floor(P/VOX).astype(np.int64)
_, idx = np.unique(q, axis=0, return_index=True)
P, Cc = P[idx], Cc[idx]
print(f'{len(P)} after {VOX} voxel downsample')

out = f'experiments/{exp}/{exp}_dense2.ply'
with open(out,'w') as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, c in zip(P, Cc):
        f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
print('wrote', out)
