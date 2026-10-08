# Surface reconstruction from OneSLAM sparse maps on endoscopic video

Running [OneSLAM](https://github.com/arcadelab/OneSLAM) (Teufel et al., IJCARS
2024) on a 31-second laparoscopic sequence, then turning its sparse point cloud
into a surface two ways — alpha shapes (Delaunay) and Poisson — and comparing
what each produces from the same noisy input.

Two things came out of it that are worth more than the meshes: the CoTracker
window size the paper specifies silently destroys this sequence, and a
trajectory coherence check is a prerequisite to looking at any reconstruction
at all.

## The sequence

A 31-second window (1:07–1:37) of laparoscopic video, 932 frames extracted at
30 fps, 640×540. Extraction at 30 rather than 10 fps is deliberate — at 10 fps
inter-frame flow on this footage runs ~11.5 px and exhausts the tracker.

Assumed intrinsics, no calibration performed:

```json
{"fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 270.0, "fps": 30}
```

## Reproducing

OneSLAM carries no license file, so its source is not redistributed here.

```bash
git clone --recursive https://github.com/arcadelab/OneSLAM.git
cd OneSLAM
git checkout 05c1543
git -C submodules/cotracker checkout 8d36403   # CoTracker v1.0
```

Extract frames, then run:

```bash
python run_slam.py \
  --data_root <sequence> \
  --point_sampler sift \
  --tracked_point_num_max 6000 \
  --minimum_new_points 300 \
  --point_resample_cooldown 0 \
  --keyframe_subsample 2 \
  --cotracker_model cotracker_stride_4_wind_12 \
  --cotracker_window_size 12 \
  --ransac_localization \
  --update_localized_pose \
  --name ep_0107_w12
```

Both `--cotracker_model` and `--cotracker_window_size` must be set together;
they are separate arguments and the second is not inferred from the first.

Then gate the trajectory, and only then mesh:

```bash
python scripts/coherence_check.py experiments/ep_0107_w12/poses_pred.txt
python scripts/mesh_poisson.py  experiments/ep_0107_w12/vis/points.ply out/
python scripts/mesh_delaunay.py experiments/ep_0107_w12/vis/points.ply out/
```

Throughput on an NVIDIA A100 80GB for the w12 run: 1.02 FPS overall, 4.36 FPS
tracking, 0.12 FPS mapping.

## Limitations

- **Intrinsics are assumed, not calibrated.** Reconstruction scale and any
  metric interpretation are unreliable; the surface is qualitative.
- **No ground truth.** `poses_gt.txt` holds identity poses to satisfy the data
  loader. Any ATE or RPE the pipeline prints against it is meaningless, and the
  coherence ratio is a self-consistency check, not an accuracy measure.
- **Normal orientation is unverified.** Poisson needs oriented normals;
  `orient_normals_consistent_tangent_plane` can flip on sparse data, which would
  leave parts of the mesh locally inverted without that being obvious.
- **Divergence is gauge-dependent.** OneSLAM coordinates sit in an arbitrary
  frame, which is why alpha is swept in multiples of nearest-neighbour distance
  rather than absolute units.
- **The sampler was not ablated here.** `sift` was used throughout on the
  reasoning that this is specular tissue; `uniform` was not tried on the w12
  configuration, so the window-size result is isolated but the sampler choice
  is not.
- **One sequence, one object.** Nothing here establishes general behaviour.

## Contents

```
scripts/     coherence gate, Delaunay and Poisson meshing
configs/     camera intrinsics used
results/     point cloud, trajectories, captured run arguments
meshes/      alpha-shape sweep and depth-7 Poisson (raw and trimmed)
```

`ep_0107_w8_failure` has no `args.json`; it predates the run-configuration
patch. Its configuration is the command above with window 8.

## Citing OneSLAM

Teufel T, Shu H, Soberanis-Mukul RD, Mangulabnan JE, Sahu M, Vedula SS, Ishii M,
Hager G, Taylor RH, Unberath M. *OneSLAM to map them all: a generalized approach
to SLAM for monocular endoscopic imaging based on tracking any point.*
International Journal of Computer Assisted Radiology and Surgery,
19:1259–1266, 2024. https://doi.org/10.1007/s11548-024-03171-6

## License

Scripts and documentation: see [LICENSE](LICENSE). OneSLAM is not redistributed
here and is not covered by it. See [docs/data-provenance.md](docs/data-provenance.md)
regarding the source video and everything derived from it.
