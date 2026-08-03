"""Throwaway benchmark harness for the C++ integration feasibility spike (plan.md).

NOT part of the pytest suite. Run directly:

    python cluster_visualization/tests/bench_native_candidates.py

Compares:
  1. Baseline: existing per-pixel Python loop calling hp.boundaries()/hp.vec2ang()
     once per HEALPix pixel (as in mermosaic.py _create_grouped_mask_traces /
     _create_binary_mask_traces).
  2. Numpy-only vectorized: single batch hp.boundaries(nside, pix_array, ...)
     call (healpy's native `_boundaries_multiple` C-level loop) + single
     hp.vec2ang() call over all points at once.

Also checks shapely's vectorized `intersects()` predicate as a substitute for
the CATRED `iterrows()` loop in catred_handler.py.
"""
import time

import healpy as hp
import numpy as np
from shapely.geometry import Polygon, box
from shapely import intersects as shapely_intersects_vectorized

NSIDE = 16384
SCALES = [1_000, 5_000, 10_000, 40_000]
STEP = 2


def make_synthetic_pixels(n, seed=0):
    rng = np.random.default_rng(seed)
    npix = 12 * NSIDE * NSIDE
    pixels = rng.integers(0, npix, size=n, dtype=np.int64)
    weights = rng.uniform(0.0, 1.0, size=n)
    return pixels, weights


def baseline_loop(pixels, weights):
    """Reproduces the existing per-pixel loop body (ra/dec extraction only,
    no plotly trace construction, to isolate the healpy-call cost)."""
    all_ra = []
    all_dec = []
    for pix, weight in zip(pixels, weights):
        ra, dec = hp.vec2ang(
            hp.boundaries(NSIDE, int(pix), step=STEP, nest=True).T, lonlat=True
        )
        all_ra.append(ra)
        all_dec.append(dec)
    return all_ra, all_dec


def vectorized_batch(pixels, weights):
    """Single batch boundaries() call + single vec2ang() call for all pixels."""
    bounds = hp.boundaries(NSIDE, pixels, step=STEP, nest=True)  # (npix, 3, 4*step)
    npix, _, npts = bounds.shape
    flat = np.moveaxis(bounds, 1, 2).reshape(-1, 3)  # (npix*npts, 3)
    ra_flat, dec_flat = hp.vec2ang(flat, lonlat=True)
    ra = ra_flat.reshape(npix, npts)
    dec = dec_flat.reshape(npix, npts)
    return ra, dec


def check_correctness(pixels):
    """Confirm vectorized output matches baseline within float tolerance."""
    base_ra, base_dec = baseline_loop(pixels, np.ones(len(pixels)))
    vec_ra, vec_dec = vectorized_batch(pixels, np.ones(len(pixels)))
    base_ra_arr = np.stack(base_ra)
    base_dec_arr = np.stack(base_dec)
    ra_ok = np.allclose(base_ra_arr, vec_ra, atol=1e-9)
    dec_ok = np.allclose(base_dec_arr, vec_dec, atol=1e-9)
    return ra_ok and dec_ok


def bench_healpix():
    print("=== HEALPix mask pixel-boundary loop ===")
    print(f"{'n_pixels':>10} {'baseline (s)':>14} {'vectorized (s)':>16} {'speedup':>9}")
    for n in SCALES:
        pixels, weights = make_synthetic_pixels(n)

        t0 = time.perf_counter()
        baseline_loop(pixels, weights)
        t_baseline = time.perf_counter() - t0

        t0 = time.perf_counter()
        vectorized_batch(pixels, weights)
        t_vec = time.perf_counter() - t0

        speedup = t_baseline / t_vec if t_vec > 0 else float("inf")
        print(f"{n:>10} {t_baseline:>14.4f} {t_vec:>16.4f} {speedup:>8.1f}x")

    correct = check_correctness(make_synthetic_pixels(2_000)[0])
    print(f"\nCorrectness (np.allclose vs baseline, n=2000 pixels): {correct}")


def make_synthetic_candidates(n=40, seed=1):
    rng = np.random.default_rng(seed)
    polys = []
    for _ in range(n):
        cx, cy = rng.uniform(-10, 10, size=2)
        w, h = rng.uniform(0.05, 0.5, size=2)
        polys.append(box(cx - w, cy - h, cx + w, cy + h))
    zoom_box = box(-5, -5, 5, 5)
    return polys, zoom_box


def bench_catred():
    print("\n=== CATRED tile intersection loop ===")
    polys, zoom_box = make_synthetic_candidates()

    t0 = time.perf_counter()
    baseline_mask = [poly.intersects(zoom_box) for poly in polys]
    t_baseline = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_mask = shapely_intersects_vectorized(np.array(polys, dtype=object), zoom_box)
    t_vec = time.perf_counter() - t0

    match = list(vec_mask) == baseline_mask
    print(f"iterrows-style loop: {t_baseline:.6f}s")
    print(f"shapely.intersects() vectorized: {t_vec:.6f}s")
    print(f"Boolean mask matches baseline exactly: {match}")


if __name__ == "__main__":
    bench_healpix()
    bench_catred()
