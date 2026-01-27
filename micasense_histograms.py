#!/usr/bin/env python3
"""
micasense_histograms.py

- Search: UAV/<Site>/**/dualcamera/**/(blue|red)/**.tif(f)
- Band parsed from filename: IMG_001_6.tif -> band=6
- Split into MAX 2 flights using GPS altitude (EXIF GPSAltitude via exifread):
    baseline = median altitude of first N samples with altitude
    start flight when alt >= baseline + threshold
    end flight when alt <= (baseline + threshold - hysteresis)
  -> Detect up to 2 flight segments, Flight 01 and Flight 02.
- Also compute JOIN histograms using ALL images (both flights together)
- Output per-camera histograms (blue/red) BUT:
    NO subfolders for camera, everything inside flight folders.

Outputs:
UAV/histogram/<Site>/flight_01/band_06_blue.png
UAV/histogram/<Site>/flight_01/band_06_red.png
UAV/histogram/<Site>/flight_02/band_06_blue.png
UAV/histogram/<Site>/flight_02/band_06_red.png
UAV/histogram/<Site>/join/band_06_blue.png
UAV/histogram/<Site>/join/band_06_red.png

Logs:
UAV/histogram/<Site>/flight_summary.txt
UAV/histogram/<Site>/bad_tiffs.txt
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, date, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import tifffile as tiff
    from tifffile import TiffFileError
except ImportError as e:
    raise SystemExit("Missing dependency 'tifffile'. Install with: pip install tifffile") from e

try:
    import exifread  # pip install exifread
except ImportError as e:
    raise SystemExit("Missing dependency 'exifread'. Install with: pip install exifread") from e

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None


BAND_RE = re.compile(r"_(\d+)\.(tif|tiff)$", re.IGNORECASE)

TIFF_HEADERS = {
    b"II*\x00",  # TIFF LE
    b"MM\x00*",  # TIFF BE
    b"II+\x00",  # BigTIFF LE
    b"MM\x00+",  # BigTIFF BE
}

RE_EXIF_DT = re.compile(r"(\d{4}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2})")


@dataclass(frozen=True)
class ImgRec:
    path: Path
    band: int
    camera: str           # 'blue' | 'red'
    t: datetime           # capture time (fallback: file mtime)
    alt: Optional[float]  # meters (GPSAltitude); may be None


def parse_band_number(filename: str) -> Optional[int]:
    m = BAND_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def classify_camera(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "blue" in parts:
        return "blue"
    if "red" in parts:
        return "red"
    return "unknown"


def is_in_dualcamera(path: Path) -> bool:
    return any(p.lower() == "dualcamera" for p in path.parts)


def looks_like_tiff(fp: Path) -> bool:
    try:
        if fp.stat().st_size < 8:
            return False
        with fp.open("rb") as f:
            header = f.read(4)
        return header in TIFF_HEADERS
    except Exception:
        return False


def find_micasense_tifs(uav_root: Path, site: str) -> List[Path]:
    site_dir = uav_root / site
    if not site_dir.exists():
        raise FileNotFoundError(f"Site folder not found: {site_dir}")

    out: List[Path] = []
    for p in site_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".tif", ".tiff"}:
            continue
        if not is_in_dualcamera(p):
            continue
        cam = classify_camera(p)
        if cam in {"blue", "red"}:
            out.append(p)
    return sorted(out)


def iter_with_progress(items: List, desc: str) -> Iterable:
    n = len(items)
    if n == 0:
        return iter(())
    if tqdm is not None:
        return tqdm(items, desc=desc, unit="file", dynamic_ncols=True)

    def _gen():
        last = -1
        for i, it in enumerate(items, 1):
            pct = int(i * 100 / n)
            if pct != last:
                print(f"\r{desc}: {pct:3d}% ({i}/{n})", end="", flush=True)
                last = pct
            yield it
        print()

    return _gen()


def _ratio_to_float(v) -> Optional[float]:
    try:
        if hasattr(v, "num") and hasattr(v, "den"):
            if v.den == 0:
                return None
            return float(v.num) / float(v.den)
        if isinstance(v, (int, float)):
            return float(v)
    except Exception:
        return None
    return None


def extract_alt_time_exifread(fp: Path) -> Tuple[Optional[float], Optional[datetime]]:
    """
    Read GPS altitude and capture time using exifread from TIFF.
    """
    try:
        with fp.open("rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return None, None

    # altitude
    alt = None
    alt_tag = tags.get("GPS GPSAltitude")
    if alt_tag is not None:
        try:
            val = alt_tag.values
            if isinstance(val, list) and len(val) > 0:
                alt = _ratio_to_float(val[0])
            else:
                alt = _ratio_to_float(val)
        except Exception:
            alt = None

        ref_tag = tags.get("GPS GPSAltitudeRef")
        if alt is not None and ref_tag is not None:
            try:
                ref_vals = ref_tag.values
                ref = ref_vals[0] if isinstance(ref_vals, list) and ref_vals else ref_vals
                if int(ref) == 1:
                    alt = -alt
            except Exception:
                pass

    # time
    dt = None
    for k in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
        tag = tags.get(k)
        if tag is None:
            continue
        try:
            s = str(tag.values)
            m = RE_EXIF_DT.search(s)
            if m:
                dt = datetime.strptime(m.group(1), "%Y:%m:%d %H:%M:%S")
                break
        except Exception:
            continue

    # GPS time fallback
    if dt is None:
        try:
            dtag = tags.get("GPS GPSDateStamp")
            ttag = tags.get("GPS GPSTimeStamp")
            if dtag is not None and ttag is not None:
                dstr = str(dtag.values).replace("-", ":")
                yyyy, mm, dd = [int(x) for x in dstr.split(":")[:3]]
                d = date(yyyy, mm, dd)

                tv = ttag.values
                h = _ratio_to_float(tv[0]) if len(tv) > 0 else None
                m_ = _ratio_to_float(tv[1]) if len(tv) > 1 else None
                s_ = _ratio_to_float(tv[2]) if len(tv) > 2 else None
                if h is not None and m_ is not None and s_ is not None:
                    tt = time(int(h), int(m_), int(s_))
                    dt = datetime.combine(d, tt)
        except Exception:
            pass

    return alt, dt


def update_hist(counts: np.ndarray, img: np.ndarray, bins: int, value_range: Tuple[int, int], stride: int) -> None:
    if img.ndim == 2:
        data = img[::stride, ::stride].reshape(-1)
    else:
        data = img.reshape(-1)
        if stride > 1:
            data = data[::stride]

    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.floating):
        data = data[np.isfinite(data)]

    hist, _ = np.histogram(data, bins=bins, range=value_range)
    counts += hist.astype(np.int64)


def save_hist_plot(out_path: Path, counts: np.ndarray, bins: int, value_range: Tuple[int, int], title: str) -> None:
    lo, hi = value_range
    edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0

    plt.figure()
    plt.plot(centers, counts)
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.yscale("log")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_bands(spec: str) -> List[int]:
    spec = spec.strip()
    if "," not in spec and "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))

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


def detect_flight_segments_by_altitude(
    recs_sorted: List[ImgRec],
    baseline_first_n: int,
    alt_threshold_m: float,
    hysteresis_m: float,
    min_segment_len: int,
) -> Tuple[int, Optional[float], Optional[datetime]]:
    """
    Detect up to 2 flights using altitude state machine.
    Returns:
      n_flights (1 or 2),
      baseline altitude,
      split_time (start time of flight 2) or None
    """
    alts: List[float] = []
    for r in recs_sorted:
        if r.alt is not None:
            alts.append(r.alt)
        if len(alts) >= baseline_first_n:
            break

    if len(alts) < max(10, baseline_first_n // 3):
        return 1, None, None

    baseline = float(np.median(np.array(alts, dtype=np.float64)))
    start_thr = baseline + float(alt_threshold_m)
    stop_thr = start_thr - float(hysteresis_m)

    segments: List[Tuple[int, int]] = []
    in_flight = False
    seg_start = None

    for i, r in enumerate(recs_sorted):
        alt = r.alt
        if alt is None:
            continue

        if not in_flight and alt >= start_thr:
            in_flight = True
            seg_start = i
        elif in_flight and alt <= stop_thr:
            in_flight = False
            if seg_start is not None:
                segments.append((seg_start, i + 1))
            seg_start = None

    if in_flight and seg_start is not None:
        segments.append((seg_start, len(recs_sorted)))

    good = [(s, e) for (s, e) in segments if (e - s) >= min_segment_len]
    if not good:
        return 1, baseline, None
    if len(good) == 1:
        return 1, baseline, None

    split_time = recs_sorted[good[1][0]].t
    return 2, baseline, split_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="Site folder name, e.g. Site11")
    ap.add_argument("--uav-root", default=None, help="UAV root (default: folder where this script lives)")
    ap.add_argument("--bins", type=int, default=2048, help="Histogram bin count (default 2048)")
    ap.add_argument("--range", default="0,65535", help="Histogram range 'min,max' (default 0,65535)")
    ap.add_argument("--stride", type=int, default=1, help="Subsample stride (default 1)")
    ap.add_argument("--bands", default="1-10", help="Bands: '1-10' or '1,2,6'")
    ap.add_argument("--baseline-first-n", type=int, default=80, help="How many early altitude samples to estimate baseline (default 80)")
    ap.add_argument("--alt-threshold-m", type=float, default=30.0, help="Altitude threshold above baseline to mark flight (default 30m)")
    ap.add_argument("--hysteresis-m", type=float, default=10.0, help="Hysteresis to end flight (default 10m)")
    ap.add_argument("--min-segment-len", type=int, default=80, help="Minimum consecutive in-flight images (default 80)")
    args = ap.parse_args()

    uav_root = Path(args.uav_root).resolve() if args.uav_root else Path(__file__).resolve().parent
    site = args.site
    wanted_bands = parse_bands(args.bands)

    # range
    try:
        rmin_s, rmax_s = args.range.split(",")
        value_range = (int(rmin_s.strip()), int(rmax_s.strip()))
        if value_range[0] >= value_range[1]:
            raise ValueError
    except Exception:
        raise SystemExit("Invalid --range. Use format like: --range 0,65535")

    bins = int(args.bins)
    stride = max(1, int(args.stride))

    base_out = uav_root / "histogram" / site
    base_out.mkdir(parents=True, exist_ok=True)

    bad_log = base_out / "bad_tiffs.txt"
    bad_log.write_text("reason\tpath\textra\n-----\t----\t-----\n")

    summary_txt = base_out / "flight_summary.txt"

    print(f"UAV root: {uav_root}")
    print(f"Site: {site}")
    print(f"Bands: {wanted_bands}")
    print(f"Bins: {bins}, Range: {value_range}, Stride: {stride}")
    print(f"Altitude split: baseline_first_n={args.baseline_first_n}, threshold={args.alt_threshold_m}m, hysteresis={args.hysteresis_m}m, min_segment={args.min_segment_len}")
    print(f"Output: {base_out}")
    print()

    tifs = find_micasense_tifs(uav_root, site)
    if not tifs:
        raise SystemExit(f"No TIFFs found under {uav_root/site} that match dualcamera/(blue|red).")

    records: List[ImgRec] = []
    bad_count = 0

    print(f"Found {len(tifs)} candidate TIFFs. Reading EXIF (time + GPS altitude)...")
    for fp in iter_with_progress(tifs, desc="Scan"):
        band = parse_band_number(fp.name)
        cam = classify_camera(fp)
        if band is None or band not in wanted_bands:
            continue
        if cam not in {"blue", "red"}:
            continue

        if not looks_like_tiff(fp):
            bad_count += 1
            with bad_log.open("a") as f:
                f.write(f"NOT_TIFF_OR_EMPTY\t{fp}\t-\n")
            continue

        alt, dt = extract_alt_time_exifread(fp)
        if dt is None:
            dt = datetime.fromtimestamp(fp.stat().st_mtime)

        records.append(ImgRec(path=fp, band=band, camera=cam, t=dt, alt=alt))

    if not records:
        raise SystemExit("No valid images after filtering and skipping bad files.")

    records_sorted = sorted(records, key=lambda r: r.t)
    alt_available = sum(1 for r in records_sorted if r.alt is not None)

    n_flights, baseline_alt, split_time = detect_flight_segments_by_altitude(
        records_sorted,
        baseline_first_n=int(args.baseline_first_n),
        alt_threshold_m=float(args.alt_threshold_m),
        hysteresis_m=float(args.hysteresis_m),
        min_segment_len=int(args.min_segment_len),
    )

    def flight_of(r: ImgRec) -> int:
        if n_flights == 1 or split_time is None:
            return 1
        return 1 if r.t < split_time else 2

    # Groups:
    #   flight_01, flight_02, and join (all images)
    # Key: (group_name, camera, band) -> list[Path]
    groups: Dict[Tuple[str, str, int], List[Path]] = {}

    for r in records_sorted:
        fl = flight_of(r)
        groups.setdefault((f"flight_{fl:02d}", r.camera, r.band), []).append(r.path)
        groups.setdefault(("join", r.camera, r.band), []).append(r.path)

    # Summary
    lines = []
    lines.append(f"Site: {site}")
    lines.append(f"Valid files used: {len(records_sorted)}")
    lines.append(f"Bad/empty/non-TIFF skipped: {bad_count}")
    lines.append(f"Altitude available: {alt_available}/{len(records_sorted)}")
    if baseline_alt is None:
        lines.append("Baseline altitude: N/A (not enough altitude; treated as 1 flight)")
    else:
        start_thr = baseline_alt + float(args.alt_threshold_m)
        stop_thr = start_thr - float(args.hysteresis_m)
        lines.append(f"Baseline altitude (median early): {baseline_alt:.3f} m")
        lines.append(f"Start flight threshold: {start_thr:.3f} m")
        lines.append(f"End flight threshold:   {stop_thr:.3f} m")
    lines.append(f"Detected flights: {n_flights}")
    if split_time is not None:
        lines.append(f"Flight 2 starts at: {split_time.isoformat(sep=' ')}")

    for fl in range(1, n_flights + 1):
        ts = [r.t for r in records_sorted if flight_of(r) == fl]
        ts.sort()
        if ts:
            lines.append(f"Flight {fl:02d}: {ts[0].isoformat(sep=' ')} -> {ts[-1].isoformat(sep=' ')} | files={len(ts)}")

    summary_txt.write_text("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print()

    # Create output dirs (ONLY flight folders + join)
    (base_out / "join").mkdir(parents=True, exist_ok=True)
    for fl in range(1, n_flights + 1):
        (base_out / f"flight_{fl:02d}").mkdir(parents=True, exist_ok=True)

    # Compute histograms:
    # For each band, you will get:
    #   flight_01: band_XX_blue.png / band_XX_red.png
    #   flight_02: ...
    #   join: ...
    group_names = ["flight_01"]
    if n_flights == 2:
        group_names.append("flight_02")
    group_names.append("join")

    for group in group_names:
        for cam in ("blue", "red"):
            for band in wanted_bands:
                files = groups.get((group, cam, band), [])
                if not files:
                    continue

                counts = np.zeros(bins, dtype=np.int64)
                processed = 0
                skipped = 0

                desc = f"{group} {cam} B{band:02d}"
                for fp in iter_with_progress(sorted(files), desc=desc):
                    try:
                        if not looks_like_tiff(fp):
                            skipped += 1
                            with bad_log.open("a") as f:
                                f.write(f"NOT_TIFF_OR_EMPTY\t{fp}\t-\n")
                            continue

                        img = tiff.imread(str(fp))
                        update_hist(counts, img, bins=bins, value_range=value_range, stride=stride)
                        processed += 1

                    except (TiffFileError, OSError, ValueError) as e:
                        skipped += 1
                        with bad_log.open("a") as f:
                            f.write(f"READ_ERROR\t{fp}\t{repr(e)}\n")
                        continue

                out_png = base_out / group / f"band_{band:02d}_{cam}.png"
                save_hist_plot(
                    out_png,
                    counts,
                    bins=bins,
                    value_range=value_range,
                    title=f"{site} | {group} | {cam} | Band {band:02d} | files={processed} (skipped={skipped})",
                )

    print("Done.")
    print(f"Summary: {summary_txt}")
    print(f"Bad files log: {bad_log}")


if __name__ == "__main__":
    main()
