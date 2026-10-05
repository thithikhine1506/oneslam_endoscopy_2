#!/usr/bin/env python3
"""Poisson surface reconstruction from a OneSLAM point cloud.

    python mesh_poisson.py experiments/<run>/vis/points.ply [outdir]

Uses the same filtering as mesh_delaunay.py so the two are comparable.
Writes both raw and density-trimmed meshes: the trimmed versions drop the
low-confidence regions where Poisson interpolated across gaps in the data.
Comparing raw against trimmed shows how much of the surface was invented.

Check the trajectory with coherence_check.py before meshing.
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

P = np.asarray(pcd.points)
c = np.median(P, axis=0)
d = np.linalg.norm(P - c, axis=1)
pcd = pcd.select_by_index(np.where(d < np.percentile(d, 97))[0])
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
print('after filtering: %d points' % len(pcd.points))

nn = np.median(pcd.compute_nearest_neighbor_distance())
print('median nearest-neighbour distance: %.4f' % nn)

# Poisson needs oriented normals; Delaunay does not. Orientation can flip on
# sparse data - if the mesh looks inside-out, that is the likely cause.
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=nn * 10, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(30)
print('normals estimated and oriented')

o3d.io.write_point_cloud(os.path.join(outdir, 'poisson_input_with_normals.ply'), pcd)

for depth in [6, 7, 8]:
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=False)
    dens = np.asarray(dens)

    raw = len(mesh.triangles)
    o3d.io.write_triangle_mesh(os.path.join(outdir, 'poisson_d%d_raw.ply' % depth), mesh)

    for q in [0.05, 0.15]:
        m = o3d.geometry.TriangleMesh(mesh)
        m.remove_vertices_by_mask(dens < np.quantile(dens, q))
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
        out = os.path.join(outdir, 'poisson_d%d_trim%02d.ply' % (depth, int(q * 100)))
        o3d.io.write_triangle_mesh(out, m)
        print('depth=%d raw=%6d  trim@%2d%% -> %6d triangles  %s'
              % (depth, raw, int(q * 100), len(m.triangles), out))
