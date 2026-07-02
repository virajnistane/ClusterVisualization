#!/usr/bin/env python3
"""
Generate an Aladin-compatible HiPS tile tree from the corrected mask FITS file.

PNGs are inverted so masked regions appear white with alpha encoding, suitable
for display as a HiPS image survey in Aladin Lite.

Usage:
    # Read paths from config file:
    python generate_mask_hips.py --config path/to/config.ini [--aladin-jar Aladin.jar]

    # Provide paths directly:
    python generate_mask_hips.py --fits mask.fits --output hips_out/ [--aladin-jar Aladin.jar]

After generation, set [paths] corrected_mask_hips = <output_dir> in your config.ini.
"""

import argparse
import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from PIL import Image, ImageOps


def _progress(items, desc=""):
    if tqdm is not None:
        return tqdm(list(items), desc=desc)
    items = list(items)
    print(f"{desc}: {len(items)} files")
    return items


def _find_aladin_jar(explicit_path=None):
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_file():
            sys.exit(f"Error: Aladin.jar not found at {explicit_path}")
        return p
    env_path = os.environ.get("ALADIN_JAR")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
    for name in ("Aladin.jar", "AladinBeta.jar"):
        cwd_jar = Path.cwd() / name
        if cwd_jar.is_file():
            return cwd_jar
        script_jar = Path(__file__).parent / name
        if script_jar.is_file():
            return script_jar
    sys.exit(
        "Error: Aladin.jar not found. Provide --aladin-jar, set ALADIN_JAR env var, "
        "or place Aladin.jar in the current directory."
    )


def _read_config(config_path):
    cfg = configparser.ConfigParser(interpolation=configparser.BasicInterpolation())
    cfg.read(config_path)

    def _get(section, key):
        if cfg.has_option(section, key):
            return cfg.get(section, key).strip() or None
        return None

    fits_path = _get("paths", "corrected_mask_fits")
    hips_out = _get("paths", "corrected_mask_hips")

    if not fits_path:
        sys.exit("Error: [paths] corrected_mask_fits is missing or empty in config.")
    if not hips_out:
        sys.exit(
            "Error: [paths] corrected_mask_hips is missing or empty in config.\n"
            "Set it to the desired output directory for the HiPS tiles."
        )
    return Path(fits_path), Path(hips_out)


def _resolve_output_dir(hips_out):
    """Return hips_out unchanged if empty/non-existent, else hips_out/hipsgen_output."""
    if hips_out.exists() and any(hips_out.iterdir()):
        actual = hips_out / "hipsgen_output"
        print(f"Output dir is not empty — writing to: {actual}")
        return actual
    return hips_out


def _validate(fits_path, hips_out):
    if not fits_path.is_file():
        sys.exit(f"Error: FITS file not found: {fits_path}")
    parent = hips_out.parent
    if not parent.exists():
        sys.exit(f"Error: Output parent directory does not exist: {parent}")
    if shutil.which("java") is None:
        sys.exit("Error: 'java' not found on PATH. Install Java to run hipsgen.")


def _run_hipsgen(aladin_jar, fits_path, hips_out):
    """Run hipsgen from FITS parent using relative paths (mirrors create.sh), then move output.

    hipsgen requires a relative out= path in its CWD to work correctly. We use a temp sibling
    dir next to the FITS, then move it to the requested destination. The temp dir is always
    cleaned up — on success after the move, on failure before exit.
    """
    stem = fits_path.stem
    tmp_out = fits_path.parent / stem
    if tmp_out.exists():
        print(f"Removing stale temp dir: {tmp_out}")
        shutil.rmtree(tmp_out)

    cmd = [
        "java", "-jar", str(aladin_jar.resolve()),
        "-hipsgen",
        f"in={fits_path.name}",
        f"out={stem}",
        f"id={stem}",
    ]
    print(f"Running (cwd={fits_path.parent}): {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=fits_path.parent)
        if result.returncode != 0:
            if not tmp_out.exists():
                sys.exit(f"Error: hipsgen failed (exit {result.returncode}) and produced no output.")
            print(f"Warning: hipsgen exited {result.returncode} but output exists — continuing.")

        if not tmp_out.exists():
            sys.exit("Error: hipsgen produced no output directory.")

        if tmp_out.resolve() != hips_out.resolve():
            if hips_out.exists():
                shutil.rmtree(hips_out)
            shutil.move(str(tmp_out), str(hips_out))
            print(f"Moved {tmp_out} -> {hips_out}")
            tmp_out = None  # successfully moved — nothing to clean up
    finally:
        if tmp_out is not None and tmp_out.exists():
            print(f"Cleaning up temp dir: {tmp_out}")
            shutil.rmtree(tmp_out)


def _remove_fits_tiles(hips_out):
    fits_tiles = list(hips_out.rglob("*.fits"))
    print(f"Removing {len(fits_tiles)} FITS tile(s)...")
    for f in fits_tiles:
        f.unlink()


def _fix_properties(hips_out):
    props = hips_out / "properties"
    if not props.is_file():
        print("Warning: properties file not found — skipping hips_tile_format patch.")
        return
    lines = props.read_text().splitlines()
    patched = []
    for line in lines:
        if line.strip().startswith("hips_tile_format"):
            patched.append("hips_tile_format     = png")
        else:
            patched.append(line)
    props.write_text("\n".join(patched) + "\n")
    print("Updated hips_tile_format = png in properties.")


def _invert_pngs(hips_out):
    pngs = list(hips_out.rglob("*.png"))
    print(f"Inverting {len(pngs)} PNG tile(s)...")
    for f in _progress(pngs, desc="Inverting PNGs"):
        alpha = Image.open(f).convert("L")
        alpha = ImageOps.invert(alpha)
        white = Image.new("L", alpha.size, 255)
        out = Image.merge("LA", (white, alpha))
        out.save(f, optimize=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate inverted HiPS tiles from corrected mask FITS for Aladin display."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", metavar="CONFIG_INI",
                     help="Config .ini file; reads corrected_mask_fits and corrected_mask_hips from [paths].")
    src.add_argument("--fits", metavar="FITS_FILE",
                     help="Direct path to corrected mask FITS file.")
    parser.add_argument("--output", metavar="OUTPUT_DIR",
                        help="Output directory for HiPS tiles (required with --fits).")
    parser.add_argument("--aladin-jar", metavar="ALADIN_JAR",
                        help="Path to Aladin.jar. Defaults to ALADIN_JAR env var or Aladin.jar in CWD.")
    args = parser.parse_args()

    if args.fits and not args.output:
        parser.error("--output is required when using --fits.")

    if args.config:
        fits_path, hips_out = _read_config(args.config)
    else:
        fits_path = Path(args.fits)
        hips_out = Path(args.output)

    fits_path = fits_path.resolve()
    hips_out = hips_out.resolve()
    hips_out = _resolve_output_dir(hips_out)
    _validate(fits_path, hips_out)
    aladin_jar = _find_aladin_jar(args.aladin_jar)

    print(f"FITS input : {fits_path}")
    print(f"HiPS output: {hips_out}")
    print(f"Aladin.jar : {aladin_jar}")
    print()

    _run_hipsgen(aladin_jar, fits_path, hips_out)
    _remove_fits_tiles(hips_out)
    _fix_properties(hips_out)
    _invert_pngs(hips_out)

    print()
    print(f"Done. HiPS tiles at: {hips_out}")
    if args.config:
        print(f"Verify [paths] corrected_mask_hips = {hips_out} is set in {args.config}")
    else:
        print(f"Set [paths] corrected_mask_hips = {hips_out} in your config.ini")


if __name__ == "__main__":
    main()
