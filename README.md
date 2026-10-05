# Surface reconstruction from OneSLAM sparse maps on endoscopic video

Running [OneSLAM](https://github.com/arcadelab/OneSLAM) (Teufel et al., IJCARS
2024) on a 31-second laparoscopic sequence, then turning its sparse point cloud
into a surface two ways — alpha shapes (Delaunay) and Poisson — and comparing
what each produces from the same noisy input.

Two things came out of it that are worth more than the meshes: the CoTracker
window size the paper specifies silently destroys this sequence, and a
trajectory coherence check is a prerequisite to looking at any reconstruction
at all.

## Findings

### 1. CoTracker window size decides whether the run is usable

The paper states its configuration as stride 4, window 8, without qualification.
On this sequence that produces a trajectory with 26 teleports; window 12, with
every other argument identical, produces none.

| Run | CoTracker window | Keyframes | Points | Coherence ratio | Spikes |
|---|---|---|---|---|---|
| `ep_0107_w8_failure` | 8 | 463 | 20,226 | **122.9** | **26** |
| `ep_0107_w12` | 12 | 463 | 20,763 | **4.3** | **0** |

The cause is measurable. Farneback optical flow across the frames where tracking
collapses:

```
150->152  median= 3.60 px      166->168  median=11.79 px
154->156  median=14.94 px      170->172  median= 4.25 px
158->160  median=18.12 px
```

(2-frame gaps, so roughly half those values per frame.) Baseline motion is
~1.8 px/frame; through frames 154–168 it rises to ~9 px/frame and stays there.
Window 8 cannot bridge it — every tracked point is lost and
`tracking.py:119` asserts out, or localization returns poses from the wrong
basin. Image quality is unchanged across the collapse (mean 72–75, std 48.5–52.8
over frames 150–180), so this is a motion limit, not a blur or exposure problem.

The paper names fast camera movement as a failure mode. This quantifies where
the threshold sits for its default configuration, on this content.

### 2. Check the trajectory before looking at the map

The failing run produced **20,226 points**; the clean run produced **20,763**.
Map size does not separate them. The trajectory does:

```
ep_0107_w8_failure   n=463  median=0.2390  p99=29.3730  ratio=122.9  spikes=26
ep_0107_w12          n=463  median=0.2175  p99= 0.9378  ratio=  4.3  spikes=0
```

A cloud built on a teleporting trajectory has points from different frames
placed in inconsistent coordinate frames. Any surface extracted from it
describes the pose error, not the anatomy. `scripts/coherence_check.py` reports
p99/median step ratio and the count of steps exceeding 10× the median; both
meshing scripts should only ever be pointed at a cloud that passes it.

### 3. Poisson suits this data; Delaunay does not

Both methods ran on the identical filtered cloud (20,763 → 19,611 points after
a 97th-percentile distance cut and statistical outlier removal; median
nearest-neighbour distance 0.0276).

**Alpha shapes** connect points into triangles wherever a ball of radius alpha
cannot pass between them, so the surface passes through every point — including
every badly triangulated one.

| alpha | Triangles | Watertight |
|---|---|---|
| 0.0553 (2× nn) | 9,182 | no |
| 0.1105 (4× nn) | 21,532 | no |
| 0.2210 (8× nn) | **31,154** | no |
| 0.4420 (16× nn) | 28,640 | no |
| 0.8841 (32× nn) | 18,440 | no |

Triangle count peaks at 8× and falls off either side: below it the ball slips
through most gaps and the shell fragments; above it distant points merge into
fewer, larger triangles spanning the real surface.

**Poisson** fits a smooth implicit surface near the points rather than through
them, and reports a per-vertex density saying how much data supported each
piece of surface.

| depth | Raw | Trim @5% | Trim @15% |
|---|---|---|---|
| 6 | 7,153 | 7,099 | 6,085 |
| 7 | 20,267 | 19,275 | 17,141 |
| 8 | 61,375 | 58,250 | 51,306 |

Depth 7 sits near one triangle per input point, which is the honest resolution
for this cloud; depth 8's 61K triangles over 19,611 points is manufactured
detail.

The density trim is the part alpha shapes has no equivalent for. Rendered side
by side, the 5%-trimmed mesh carries large pale sheets around its perimeter with
essentially no input points on them — Poisson extrapolating outward past its
last data. At 15% those are gone and what remains follows the L-shaped footprint
the scope actually swept, with roughly even point coverage throughout. Delaunay
gives no comparable signal: interpolated and well-supported geometry look
identical in its output.

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
