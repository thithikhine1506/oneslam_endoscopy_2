#!/usr/bin/env python3
"""Trajectory coherence gate for OneSLAM runs.

A reconstruction can have a large, healthy-looking point count and still be
built on a broken trajectory. Run this before interpreting any map.

Two statistics, both on inter-keyframe step length:

  ratio   p99 / median. Smooth camera motion gives a small number (~3-5).
          Large values mean some steps are wildly out of scale with the rest.

  spikes  count of steps exceeding 10x the median. Nonzero means the
          trajectory teleports. Isolated spikes usually appear in pairs
          (out and back) - a single bad pose between two good ones.

This is a self-consistency check, not an accuracy measure: a trajectory can be
perfectly smooth and still wrong. It only rules out the gross failure.

Usage:
    coherence_check.py experiments/*/poses_pred.txt
    coherence_check.py --show-spikes experiments/run/poses_pred.txt
"""

import argparse
import os
import sys

import numpy as np

SPIKE_FACTOR = 10.0


def analyse(path):
    pos = np.loadtxt(path, usecols=(1, 2, 3))
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    median = float(np.median(steps))
    p99 = float(np.percentile(steps, 99))
    return {
        "n": len(pos),
        "median": median,
        "p99": p99,
        "ratio": p99 / median if median > 0 else float("inf"),
        "spikes": np.where(steps > SPIKE_FACTOR * median)[0],
        "positions": pos,
        "steps": steps,
    }


def label(path):
    """Prefer the run directory name over the filename."""
    parent = os.path.basename(os.path.dirname(path))
    return parent if parent else path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses", nargs="+", help="poses_pred.txt file(s)")
    ap.add_argument("--show-spikes", action="store_true",
                    help="print the neighbourhood of each spike")
    args = ap.parse_args()

    results = {}
    for path in args.poses:
        try:
            results[path] = analyse(path)
        except Exception as exc:
            print("%-28s  could not read: %s" % (label(path), exc), file=sys.stderr)

    if not results:
        return 1

    for path, r in results.items():
        print("%-28s n=%4d  median=%.4f  p99=%.4f  ratio=%9.1f  spikes=%d"
              % (label(path), r["n"], r["median"], r["p99"], r["ratio"],
                 len(r["spikes"])))

    if args.show_spikes:
        for path, r in results.items():
            if not len(r["spikes"]):
                continue
            print("\n%s - %d spike(s):" % (label(path), len(r["spikes"])))
            for i in r["spikes"]:
                print("  step %d: %.4f (%.1fx median)"
                      % (i, r["steps"][i], r["steps"][i] / r["median"]))
                lo, hi = max(0, i - 2), min(r["n"], i + 4)
                for j in range(lo, hi):
                    # The step spans poses i and i+1; mark both rather than
                    # guessing which end is the bad one.
                    mark = "  <--" if j in (i, i + 1) else ""
                    print("    %4d: %s%s"
                          % (j, np.array2string(r["positions"][j], precision=3), mark))

    # Pairwise divergence, meaningful only between repeats of the SAME
    # configuration. Raw coordinates, no Umeyama alignment - part of any
    # difference may be gauge freedom rather than trajectory disagreement.
    paths = list(results)
    if len(paths) > 1:
        print("\npairwise max/mean pose difference (unaligned, same-config runs only):")
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                a, b = results[paths[i]], results[paths[j]]
                if a["n"] != b["n"]:
                    continue
                d = np.abs(a["positions"] - b["positions"])
                print("  %-22s vs %-22s  max=%.4f  mean=%.4f"
                      % (label(paths[i]), label(paths[j]), d.max(), d.mean()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
