# Data provenance — resolve before publishing

The source video for this work is laparoscopic surgical footage
(`1GP_Endoscopy.MP4`, 30 min, 1920x1080, 29.97 fps). Unlike self-recorded
material, its provenance determines what may be redistributed.

**This file is a placeholder. Fill it in, or remove the derived data, before
making the repository public.**

## What inherits the source video's restrictions

Everything below is a derivative work of that footage:

- the trimmed clip and the 932 extracted frames (not included here)
- `results/*/points.ply` — the 3D points are triangulated from those frames
- `meshes/*.ply` — surfaces fitted to those points, with colour sampled from
  the frames
- `results/*/poses_pred.txt` — the camera trajectory through the scene

Point clouds and meshes sit further from the source than raw frames do, and are
often publishable where frames are not, but that is a judgement for whoever
holds the rights, not an automatic exemption.

## What does not

- `scripts/` — written for this work
- the measurements in the README (flow rates, coherence ratios, triangle counts)
- the findings themselves

If the provenance question cannot be settled, the repository still stands as a
scripts-and-findings publication with the derived data removed.

## To fill in

- Source of the video (public dataset / collaborator / clinical source):
- Licence or data-use agreement, if any:
- Redistribution permitted for derived data (point clouds, meshes):
- Who confirmed this, and when:
