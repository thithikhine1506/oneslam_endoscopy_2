#!/usr/bin/env python3
"""Alpha-shape (Delaunay) surface reconstruction from a OneSLAM point cloud.

    python mesh_delaunay.py experiments/<run>/vis/points.ply [outdir]

Sweeps alpha as multiples of the cloud's median nearest-neighbour distance,
since OneSLAM coordinates are in an arbitrary gauge and absolute alpha values
are not comparable between runs.

Check the trajectory with coherence_check.py before meshing: a cloud built on
a spiked trajectory produces a surface of the pose error, not the anatomy.
"""
import os
import sys

import numpy as np
import open3d as o3d

src = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else '.'
os.makedirs(outdir, exist_ok=True)

pcd = o3d.io.read_point_cloud(src)
print('loaded %d points' % len(pcd.points))
if len(pcd.points) == 0:
    sys.exit('no points read from %s (file missing or not a readable PLY)' % src)

# Drop far-flung strays, then isolated points floating in empty space.
P = np.asarray(pcd.points)
c = np.median(P, axis=0)
d = np.linalg.norm(P - c, axis=1)
pcd = pcd.select_by_index(np.where(d < np.percentile(d, 97))[0])
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
print('after filtering: %d points' % len(pcd.points))

nn = np.median(pcd.compute_nearest_neighbor_distance())
print('median nearest-neighbour distance: %.4f' % nn)

for mult in [2, 4, 8, 16, 32]:
    a = nn * mult
    try:
        m = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, a)
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
        out = os.path.join(outdir, 'delaunay_alpha_%dx.ply' % mult)
        o3d.io.write_triangle_mesh(out, m)
        print('alpha=%.4f (%2dx nn): %6d triangles  watertight=%s  -> %s'
              % (a, mult, len(m.triangles), m.is_watertight(), out))
    except Exception as e:
        print('alpha=%.4f (%2dx nn): failed - %s' % (a, mult, e))
