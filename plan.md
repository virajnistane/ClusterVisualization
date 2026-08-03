# Plan: C++ Integration Feasibility Spike for Numerical Hotspots

## Context / Findings

Explored full codebase for numerical bottlenecks. Most hot paths are already
compiled-backed and fast (scipy.spatial.cKDTree spatial index: 10-100x speedup
already achieved; numpy-vectorized magnitude.py; PIL-based mosaic image
processing; zoom-gated numpy oval generation). Two genuine pure-Python loops
remain:

1. **HEALPix mask pixel-boundary loop** (highest ROI)
   - `cluster_visualization/src/mermosaic.py` `_create_grouped_mask_traces()`
     lines 1478-1490, `_create_binary_mask_traces()` lines 1541-1550, and a
     third similar loop at line 2393.
   - Per-pixel: `hp.vec2ang(hp.boundaries(16384, int(pix), step=2, nest=True).T, lonlat=True)`
     called once per HEALPix pixel in a Python `for pix, weight in zip(...)` loop.
   - Scale: 1,000-40,000 pixels per mask render depending on zoom/tile.
   - Estimated cost: ~0.5-2s for typical masks; called on every mask-overlay
     render (button click / zoom-triggered).
   - healpy's `boundaries()`/`vec2ang()` are Fortran/C backed per-call, but the
     Python-level loop over pixels adds overhead and prevents vectorized
     downstream binning.

2. **CATRED tile intersection loop** (secondary target)
   - `cluster_visualization/src/data/catred_handler.py` lines 507-512:
     `for _, row in candidates.iterrows(): ... poly.intersects(zoom_box)`
   - Already pre-filtered by a vectorized bbox pass (`_bbox_overlaps`, line 492)
     down to ~5-50 candidates before this loop runs, so absolute cost is low
     (~10-50ms). Shapely 2.0.2 (already a pinned dependency) supports
     **array-based vectorized predicates** (`shapely.intersects(array, geom)`),
     which may eliminate this loop with a pure-Python one-line change and
     zero new build complexity — must be checked before reaching for
     Cython/C++.

## Environment / Build Feasibility (already confirmed practical)

- Build backend: Hatchling (pyproject.toml), Python 3.9-3.11.
- Compiler toolchain already required and present: `healpy` and `astropy`
  compile C/Fortran at install time; `Singularity.def` explicitly installs
  `gcc g++ gfortran libcfitsio-dev pkg-config` etc.
- Local dev venv (`setup_venv.sh`) sources EDEN-3.1 via CVMFS, which already
  provides a working compiler (proven by healpy install working).
- No existing native-extension precedent in the repo (no numba/cython/pybind/
  ctypes usage) — this would be the first.
- No blocking constraint found for adding a compiled extension in either the
  Singularity container or the local venv path.

## Goal of This Spike

Produce a **benchmarked, throwaway prototype** comparing 3 approaches on the
HEALPix hotspot (and check a zero-code-native fix for the CATRED hotspot),
then report back a recommendation. **No production integration in this pass**
— decide after seeing real numbers.

## Steps

**Phase 1 — Baseline benchmark harness** (no native code yet)
1. Add a standalone benchmark script (e.g. `cluster_visualization/tests/bench_native_candidates.py`,
   not part of the pytest suite) that:
   - Generates synthetic HEALPix nest pixel-id + weight arrays at 1k/5k/10k/40k
     scale (nside=16384) matching realistic mask-render sizes.
   - Generates a synthetic CATRED candidate set (30-50 shapely Polygons +
     1 zoom-box Polygon) matching realistic post-bbox-filter scale.
   - Times the **existing, unmodified** `_create_grouped_mask_traces()` /
     `_create_binary_mask_traces()` and the existing CATRED `for row in
     candidates.iterrows()` loop as baselines.
2. Check installed `healpy` version's `boundaries()` API for native batch/array
   support (pass an array of pixel ids in one call) — if this alone closes
   the gap, no native extension is needed for the HEALPix hotspot at all.
   *(depends on nothing, do first — cheapest possible win)*
3. Check `shapely.intersects()` module-level vectorized predicate (shapely
   2.0+ ufunc form) as a substitute for the CATRED `iterrows()` loop — if it
   works, this is a one-line fix, no benchmarking spike needed for that path.
   *(parallel with step 2)*

**Phase 2 — Build 3 HEALPix prototypes** (*depends on Phase 1 step 2 showing native batching insufficient*; each variant independent/parallel)
4. **Numpy-only vectorization**: attempt to vectorize the coordinate-transform
   and weight-binning portion around the (still per-pixel) `hp.boundaries`
   call, batching everything except the unavoidable single healpy call.
5. **Cython prototype**: typed `.pyx` function that loops over the pixel
   array in C-level code, calling into HEALPix's C boundary/ang routines
   (via healpy's underlying C API or a direct reimplementation of the nest
   boundary formula), returning stacked ra/dec numpy arrays in one call.
   Build via `cythonize` + a minimal `setup.py`/`scikit-build-core` config,
   isolated from the main `pyproject.toml` build for now.
6. **pybind11/C++ prototype**: small C++ extension linking against
   `libhealpix_cxx` (already installed as a healpy build dependency) exposing
   a single batch `boundaries_and_ang(pixels[], nside, step)` function.
   Build via `pybind11` + `scikit-build-core` or a manual `Extension()`.

**Phase 3 — Compare and decide**
7. Run all prototypes + baseline through the Phase 1 harness at each data
   scale (1k/5k/10k/40k pixels). Record for each: wall-clock speedup vs
   baseline, output correctness (numerically match baseline within float
   tolerance), lines of new code / build complexity, and portability risk
   (must build cleanly in both the local EDEN venv and the Singularity
   container — verify in both).
8. Report findings back to the user: recommended approach (or "no native
   code needed" if step 2/3 alone solve it), effort estimate for
   productionizing, and any correctness caveats. Do not wire anything into
   `mermosaic.py`/`catred_handler.py` production code paths in this pass —
   that is a follow-up decision after review.

## Relevant files
- `cluster_visualization/src/mermosaic.py` — `_create_grouped_mask_traces()` (~L1461-1522), `_create_binary_mask_traces()` (~L1527-1560), third similar loop ~L2393. Target of prototypes; NOT modified in this pass.
- `cluster_visualization/src/data/catred_handler.py` — `_bbox_overlaps` (~L484), vectorized pre-filter (~L492), `iterrows()` loop (~L507-512). Check shapely vectorized predicate here.
- `cluster_visualization/tests/bench_native_candidates.py` (new) — benchmark harness, not a pytest test, throwaway/dev-only script.
- `Singularity.def` — reference for confirming compiler toolchain (gcc/g++/gfortran already present) when testing container build of prototypes.
- `cluster_visualization/scripts/setup_venv.sh` — reference for EDEN-3.1 compiler availability when testing local build.
- `pyproject.toml` — NOT modified in this pass (prototypes build standalone/isolated); only touched if/when a technology is chosen for productionization.

## Verification
1. Benchmark harness produces reproducible timing numbers for baseline vs each prototype at all 4 pixel scales.
2. Each prototype's output (ra/dec polygon coordinates) matches baseline output within float tolerance (e.g. `np.allclose`) for a fixed synthetic input — correctness before speed.
3. Each viable prototype builds successfully via `apptainer build` against `Singularity.def` (or a scratch copy) to confirm container compatibility, and via the local EDEN venv.
4. Manual: confirm `shapely.intersects()` vectorized form (if used for CATRED) returns identical boolean mask to the current `iterrows()` loop on the synthetic candidate set.

## Mosaic FITS Loading/Rendering — Extended Analysis (NOT added to scope)

Investigated separately whether this plan should extend to mosaic FITS
loading/rendering. Verdict: **no, do not extend** — no genuine pure-Python
bottleneck found in that pipeline.

- `_load_local_mosaic_fits_data()` / `_load_esa_cutout_by_mertile()`
  (`cluster_visualization/src/mermosaic.py` ~L326): reads via
  `astropy.io.fits` (cfitsio C backend), `.copy()` to plain ndarray,
  `astropy.wcs.WCS()` header parse (C-backed). I/O + gzip decompression
  bound, not CPU-loop bound.
- `_percentile_normalize()` / `_process_mosaic_image()` (~L966, ~L1089):
  `np.isfinite/np.nanpercentile/np.clip` (vectorized numpy ufuncs), early
  downsampling via numpy stride view `[::factor, ::factor]`, PIL LANCZOS
  resize (C-backed Pillow) — zero per-pixel Python loops.
- WCS 4-corner transform (`_calculate_image_bounds_direct()` ~L1399):
  `wcs_pix2world()` called once on 4 points per tile (constant cost,
  vectorized, C-backed WCSLIB) — negligible regardless of tile size.
- PNG encoding (`create_mosaic_image_trace()` ~L1658): numpy array
  flips/clip (vectorized) + Pillow libpng compression (C-backed). Already
  gives 35-90x payload reduction (14MB float32 → 150-400KB PNG), documented
  in `docs/MOSAIC_RENDERING_UPGRADE.md`.
- `cluster_visualization/scripts/generate_mask_hips.py`: offline/batch
  script, NOT in the live request path. Shells out to Java's `hipsgen` tool
  (via `subprocess`) to do the actual FITS→HiPS tiling; the only Python-side
  loop is `_invert_pngs()` — a per-file (not per-pixel) PIL invert over
  already-generated PNG tiles, I/O-bound, negligible CPU. Not a candidate.

**Conclusion**: every stage of the FITS-load-and-render path is already
backed by compiled C/Fortran libraries (cfitsio via astropy.io.fits, WCSLIB
via astropy.wcs, Pillow, numpy) and fully vectorized. A C++/Cython/Numba
extension here would add build complexity for no measurable gain. If this
path is ever suspected slow in practice, profile first with real large FITS
tiles (I/O/decompression will likely dominate, which no native-code
CPU optimization fixes) rather than assuming a compute bottleneck.

## fitsrs (github.com/cds-astro/fitsrs) — Evaluated, NOT adopted

Examined separately for server-side mosaic FITS loading. Verdict: not practical.

- Pure Rust FITS reader. Built specifically to read FITS/HiPS tiles inside
  **Aladin Lite** (web/WASM sky atlas) in the browser — motivation is
  no-C-dependency compilation to WASM, not raw speed superiority over cfitsio
  on native platforms.
- **No official Python bindings** (no PyO3/maturin package on PyPI). Adopting
  it server-side would mean writing a whole new PyO3 wrapper crate from
  scratch — a bigger, riskier undertaking than the pybind11/Cython prototypes
  already planned for the HEALPix hotspot, for an I/O path that's already
  cfitsio-backed via `astropy.io.fits` (see FITS pipeline analysis above).
- Feature gaps vs cfitsio: ASCII table extension not implemented, some
  compression paths ("H_compress, PLI0", dithering) explicitly marked
  not-well-tested by upstream.
- Would add a Rust toolchain (cargo) as a new build dependency alongside the
  existing C/C++ toolchain, for uncertain server-side gain.
- **Already indirectly relevant and already in use**: `cluster_visualization/ui/aladin_view.py`
  loads Aladin Lite JS client-side (`A.aladin()`); Aladin Lite itself bundles
  fitsrs internally for in-browser HiPS/FITS rendering. No action needed —
  ClusterViz already gets fitsrs's benefit for free via the Aladin Lite widget,
  with zero integration work on the Python side.

**Conclusion**: do not build a custom fitsrs Python binding. Server-side FITS
loading stays on `astropy.io.fits`/cfitsio (already fast, see prior section);
client-side Aladin Lite view already benefits from fitsrs indirectly.

## Decisions
- Scope limited to the two confirmed hotspots (HEALPix mask loop, CATRED tile intersection); other numerical code (spatial index, magnitude, mosaic image processing, oval generation) is already well-optimized and explicitly excluded.
- Mosaic FITS loading/rendering pipeline explicitly evaluated and excluded (see section above) — already fully compiled/vectorized, no pure-Python bottleneck exists.
- fitsrs (Rust) evaluated and excluded — no Python bindings exist, WASM/browser-oriented, already used indirectly via Aladin Lite client widget.
- This pass is prototype + benchmark only — no production wiring, no `pyproject.toml`/build-system changes, no new runtime dependency added yet.
- Cheapest possible fixes (healpy native batch API, shapely vectorized predicate) are checked first, before writing any Cython/C++ — if either resolves the bottleneck, that hotspot needs no native extension at all.
- Comparing 3 approaches (numpy-only, Cython, pybind11/C++) for the HEALPix hotspot per user's request, rather than committing to C++ upfront.
