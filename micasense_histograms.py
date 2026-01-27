#!/usr/bin/env python3
"""
micasense_histograms.py

Compute histograms per MicaSense band (1..10) from TIFF images under a given Site folder.

Search pattern (flexible, recursive):
UAV/<Site>/**/dualcamera/**/(blue|red)/**/<images>.tif(f)

Band number is parsed from filenames like: IMG_001_6.tif  -> band=6

Outputs:
UAV/histogram/<Site>/band_06_all.png
UAV/histogram/<Site>/band_06_blue.png
UAV/histogram/<Site>/band_06_red.png
(and similarly for all selected bands)

Robustness:
- Skips 0-byte / truncated / non-TIFF files (common cause of tifffile header b'')
- Logs skipped files to: UAV/histogram/<Site>/bad_tiffs.txt
- Keeps running even if some files are bad

Progress:
- Uses tqdm if installed (pip install tqdm)
- Otherwise prints a simple % progress line
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt

try:
    import tifffile as tiff
    from tifffile import TiffFileError
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'tifffile'. Install with:\n"
        "  pip install tifffile\n"
    ) from e

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None


BAND_RE = re.compile(r"_(\d+)\.(tif|tiff)$", re.IGNORECASE)

# Common TIFF / BigTIFF headers (first 4 bytes)
TIFF_HEADERS = {
    b"II*\x00",  # classic TIFF little-endian (42)
    b"MM\x00*",  # classic TIFF big-endian (42)
    b"II+\x00",  # BigTIFF little-endian (43)
    b"MM\x00+",  # BigTIFF big-endian (43)
}


def parse_band_number(filename: str) -> Optional[int]:
    """Parse band from last underscore token: IMG_001_6.tif -> 6"""
    m = BAND_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def classify_camera(path: Path) -> str:
    """Detect camera by folder name containing 'blue' or 'red' in the path."""
    parts = [p.lower() for p in path.parts]
    if "blue" in parts:
        return "blue"
    if "red" in parts:
        return "red"
    return "unknown"


def is_in_dualcamera(path: Path) -> bool:
    return any(p.lower() == "dualcamera" for p in path.parts)


def looks_like_tiff(fp: Path) -> bool:
    """
    Basic header check to avoid crashing on empty/non-tiff files.
    Also screens out tiny files (< 8 bytes).
    """
    try:
        if fp.stat().st_size < 8:
            return False
        with fp.open("rb") as f:
            header = f.read(4)
        return header in TIFF_HEADERS
    except Exception:
        return False


def find_micasense_tifs(uav_root: Path, site: str) -> List[Path]:
    """
    Find .tif/.tiff files under UAV/<site>/... but only those in a 'dualcamera' tree
    and under a 'blue' or 'red' folder somewhere in their path.
    """
    site_dir = uav_root / site
    if not site_dir.exists():
        raise FileNotFoundError(f"Site folder not found: {site_dir}")

    tifs: List[Path] = []
    for p in site_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".tif", ".tiff"}:
            continue
        if not is_in_dualcamera(p):
            continue
        cam = classify_camera(p)
        if cam in {"blue", "red"}:
            tifs.append(p)

    return sorted(tifs)


def iter_with_progress(items: List[Path], desc: str) -> Iterable[Path]:
    """Yield items with a progress indicator."""
    n = len(items)
    if n == 0:
        return iter(())

    if tqdm is not None:
        return tqdm(items, desc=desc, unit="file", dynamic_ncols=True)

    # Fallback simple % printing
    def _gen():
        last_pct = -1
        for i, it in enumerate(items, start=1):
            pct = int(i * 100 / n)
            if pct != last_pct:
                print(f"\r{desc}: {pct:3d}% ({i}/{n})", end="", flush=True)
                last_pct = pct
            yield it
        print()  # newline at end

    return _gen()


def update_hist(
    counts: np.ndarray,
    img: np.ndarray,
    bins: int,
    value_range: Tuple[int, int],
    stride: int,
) -> None:
    """
    Update histogram counts in-place using image pixels (optionally subsampled by stride).
    Works for 2D images; if image has extra dims, flattens everything.
    """
    if img.ndim == 2:
        data = img[::stride, ::stride].reshape(-1)
    else:
        # Sometimes TIFFs can be (pages, rows, cols) or similar
        data = img.reshape(-1)
        if stride > 1:
            data = data[::stride]

    data = np.asarray(data)

    # Remove NaN/inf if float
    if np.issubdtype(data.dtype, np.floating):
        data = data[np.isfinite(data)]

    hist, _ = np.histogram(data, bins=bins, range=value_range)
    counts += hist.astype(np.int64)


def save_hist_plot(
    out_path: Path,
    counts: np.ndarray,
    bins: int,
    value_range: Tuple[int, int],
    title: str,
    log_y: bool = True,
) -> None:
    lo, hi = value_range
    edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0

    plt.figure()
    plt.plot(centers, counts)
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    if log_y:
        plt.yscale("log")
        plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_bands(spec: str) -> List[int]:
    """
    Parse bands spec examples:
      '1-10'
      '1,2,3,6'
      '4-8,10'
    """
    spec = spec.strip()
    out: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="Site folder name, e.g. Site11")
    ap.add_argument(
        "--uav-root",
        default=None,
        help="Path to UAV root (default: folder where this script lives)",
    )
    ap.add_argument("--bins", type=int, default=2048, help="Histogram bin count (default: 2048)")
    ap.add_argument("--range", default="0,65535", help="Histogram value range as 'min,max' (default: 0,65535)")
    ap.add_argument("--stride", type=int, default=1, help="Subsample stride (default: 1)")
    ap.add_argument("--bands", default="1-10", help="Bands to process (default: 1-10)")
    args = ap.parse_args()

    uav_root = Path(args.uav_root).resolve() if args.uav_root else Path(__file__).resolve().parent
    site = args.site

    # Parse range
    try:
        rmin_s, rmax_s = args.range.split(",")
        value_range = (int(rmin_s.strip()), int(rmax_s.strip()))
        if value_range[0] >= value_range[1]:
            raise ValueError
    except Exception:
        raise SystemExit("Invalid --range. Use format like: --range 0,65535")

    wanted_bands = parse_bands(args.bands)
    bins = int(args.bins)
    stride = max(1, int(args.stride))

    # Output folder + log
    out_dir = uav_root / "histogram" / site
    out_dir.mkdir(parents=True, exist_ok=True)
    bad_log = out_dir / "bad_tiffs.txt"

    # Clear log each run
    bad_log.write_text(
        "reason\tpath\textra\n"
        "-----\t----\t-----\n"
    )

    # Find TIFFs
    tifs = find_micasense_tifs(uav_root, site)
    if not tifs:
        raise SystemExit(f"No TIFFs found under {uav_root/site} that match dualcamera/(blue|red).")

    # Group by (band, camera)
    files_by_band_cam: Dict[Tuple[int, str], List[Path]] = {}
    for p in tifs:
        band = parse_band_number(p.name)
        if band is None or band not in wanted_bands:
            continue
        cam = classify_camera(p)
        files_by_band_cam.setdefault((band, cam), []).append(p)

    print(f"UAV root: {uav_root}")
    print(f"Site: {site}")
    print(f"Found {len(tifs)} candidate TIFFs (blue/red under dualcamera).")
    print(f"Output: {out_dir}")
    print(f"Bins: {bins}, Range: {value_range}, Stride: {stride}")
    print(f"Bands: {wanted_bands}")
    print()

    total_bad = 0

    for band in wanted_bands:
        band_counts = {
            "blue": np.zeros(bins, dtype=np.int64),
            "red": np.zeros(bins, dtype=np.int64),
        }
        band_files = {
            "blue": files_by_band_cam.get((band, "blue"), []),
            "red": files_by_band_cam.get((band, "red"), []),
        }

        any_found = False

        for cam in ("blue", "red"):
            flist = band_files[cam]
            if not flist:
                continue

            any_found = True
            desc = f"Band {band:02d} [{cam}]"
            print(f"{desc}: {len(flist)} files")

            processed = 0
            skipped = 0

            for fp in iter_with_progress(flist, desc=desc):
                try:
                    # Quick screening to avoid crashing on empty/non-tiff
                    if not looks_like_tiff(fp):
                        skipped += 1
                        total_bad += 1
                        with bad_log.open("a") as f:
                            f.write(f"NOT_TIFF_OR_EMPTY\t{fp}\t-\n")
                        continue

                    img = tiff.imread(str(fp))
                    update_hist(band_counts[cam], img, bins=bins, value_range=value_range, stride=stride)
                    processed += 1

                except (TiffFileError, OSError, ValueError) as e:
                    skipped += 1
                    total_bad += 1
                    with bad_log.open("a") as f:
                        f.write(f"READ_ERROR\t{fp}\t{repr(e)}\n")
                    continue

            print(f"{desc}: processed={processed}, skipped={skipped} (log: {bad_log.name})")

            # Save per-camera plot (even if some skipped; if processed==0, it'll be all zeros)
            save_hist_plot(
                out_dir / f"band_{band:02d}_{cam}.png",
                band_counts[cam],
                bins=bins,
                value_range=value_range,
                title=f"{site} - Band {band:02d} ({cam})",
                log_y=True,
            )

        if not any_found:
            print(f"Band {band:02d}: no files found (skipping plots).")
            print()
            continue

        # Combined
        combined = band_counts["blue"] + band_counts["red"]
        save_hist_plot(
            out_dir / f"band_{band:02d}_all.png",
            combined,
            bins=bins,
            value_range=value_range,
            title=f"{site} - Band {band:02d} (blue+red)",
            log_y=True,
        )
        print()

    print("Done.")
    if total_bad > 0:
        print(f"WARNING: Skipped {total_bad} bad/empty/non-TIFF files. See: {bad_log}")
    else:
        print("No bad TIFFs detected.")


if __name__ == "__main__":
    main()
