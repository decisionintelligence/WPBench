import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAN_DIRS = [ARTIFACT_ROOT / "data" / "source_coordinates"]


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map normalized lowercase column names to original names."""
    return {str(c).strip().lower(): str(c) for c in df.columns}


def _find_coord_columns(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    """
    Detect coordinate columns.

    Returns:
        (coord_type, col_a, col_b)
        - coord_type in {"latlon", "xy"}
        - col_a/col_b are original column names.
    """
    norm = _normalize_columns(df)

    # Latitude/longitude preferred
    lat_keys = ["lat", "latitude"]
    lon_keys = ["lon", "lng", "longitude"]
    lat_col = next((norm[k] for k in lat_keys if k in norm), None)
    lon_col = next((norm[k] for k in lon_keys if k in norm), None)
    if lat_col and lon_col:
        return "latlon", lat_col, lon_col

    # Fallback: projected coordinates
    x_keys = ["x", "coord_x"]
    y_keys = ["y", "coord_y"]
    x_col = next((norm[k] for k in x_keys if k in norm), None)
    y_col = next((norm[k] for k in y_keys if k in norm), None)
    if x_col and y_col:
        return "xy", x_col, y_col

    return None


def _clean_coordinate_rows(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """Keep only rows with valid numeric coordinates."""
    out = df.copy()
    out[col_a] = pd.to_numeric(out[col_a], errors="coerce")
    out[col_b] = pd.to_numeric(out[col_b], errors="coerce")
    out = out.dropna(subset=[col_a, col_b]).reset_index(drop=True)
    return out


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in meters."""
    r = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def _pairwise_rel(coords: np.ndarray, coord_type: str) -> pd.DataFrame:
    """
    Build directed edge table with columns: origin_id,destination_id,cost.

    cost:
    - latlon: haversine distance (meters)
    - xy: euclidean distance (same unit as source x/y)
    """
    n = coords.shape[0]
    src = np.repeat(np.arange(n), n)
    dst = np.tile(np.arange(n), n)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    a = coords[src]
    b = coords[dst]

    if coord_type == "latlon":
        dist = _haversine_m(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    else:
        diff = a - b
        dist = np.sqrt(np.sum(diff * diff, axis=1))

    rel = pd.DataFrame(
        {
            "origin_id": src.astype(int),
            "destination_id": dst.astype(int),
            "cost": dist.astype(float),
        }
    )
    return rel


def _write_geo_rel(
    src_csv: Path,
    coord_type: str,
    col_a: str,
    col_b: str,
    output_root: Path,
) -> Tuple[Path, Path, int, int]:
    df = pd.read_csv(src_csv)
    df = _clean_coordinate_rows(df, col_a, col_b)
    if df.empty:
        raise ValueError("No valid coordinate rows after cleaning.")

    if coord_type == "latlon":
        geo_df = pd.DataFrame(
            {
                "node_id": np.arange(len(df), dtype=int),
                "Lat": df[col_a].to_numpy(dtype=float),
                "Lng": df[col_b].to_numpy(dtype=float),
            }
        )
        coords = geo_df[["Lat", "Lng"]].to_numpy()
    else:
        # x/y dataset has no true lat/lng; keep compatibility by reusing Lat/Lng columns.
        geo_df = pd.DataFrame(
            {
                "node_id": np.arange(len(df), dtype=int),
                "Lat": df[col_a].to_numpy(dtype=float),
                "Lng": df[col_b].to_numpy(dtype=float),
            }
        )
        coords = geo_df[["Lat", "Lng"]].to_numpy()

    rel_df = _pairwise_rel(coords, coord_type)

    geo_dir = output_root / "geo"
    rel_dir = output_root / "rel"
    geo_dir.mkdir(parents=True, exist_ok=True)
    rel_dir.mkdir(parents=True, exist_ok=True)

    base = src_csv.stem
    geo_path = geo_dir / f"{base}.geo"
    rel_path = rel_dir / f"{base}.rel"

    geo_df.to_csv(geo_path, index=False)
    rel_df.to_csv(rel_path, index=False)

    return geo_path, rel_path, len(geo_df), len(rel_df)


def _iter_csv_files(scan_dirs: Iterable[Path]) -> Iterable[Path]:
    for folder in scan_dirs:
        if not folder.exists():
            continue
        for p in folder.rglob("*.csv"):
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan dataset folders, detect coordinate CSVs, and generate .geo/.rel files."
    )
    parser.add_argument(
        "--scan-dir",
        action="append",
        default=None,
        help="Folder to scan (repeatable). Defaults to four dataset folders.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["sidecar", "central"],
        default="central",
        help="sidecar: write under each source folder's geo/rel; central: write under one root.",
    )
    parser.add_argument(
        "--central-output-root",
        default=str(ARTIFACT_ROOT / "data" / "forecasting" / "forecasting_wp1_all_shapes_v1"),
        help="Used only when --output-mode central.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be generated.",
    )
    args = parser.parse_args()

    scan_dirs = [Path(p) for p in args.scan_dir] if args.scan_dir else DEFAULT_SCAN_DIRS

    total_csv = 0
    converted = 0
    skipped = 0

    print("[INFO] Scan folders:")
    for d in scan_dirs:
        print(f"  - {d}")

    for csv_path in _iter_csv_files(scan_dirs):
        total_csv += 1
        try:
            df = pd.read_csv(csv_path, nrows=20)
        except Exception as e:  # pragma: no cover
            skipped += 1
            print(f"[SKIP] {csv_path} (read failed: {e})")
            continue

        found = _find_coord_columns(df)
        if found is None:
            skipped += 1
            print(f"[SKIP] {csv_path} (no coordinate columns)")
            continue

        coord_type, col_a, col_b = found

        if args.output_mode == "central":
            output_root = Path(args.central_output_root)
        else:
            output_root = csv_path.parent

        if args.dry_run:
            converted += 1
            print(
                f"[DRY-RUN] {csv_path} -> {output_root / 'geo' / (csv_path.stem + '.geo')}"
                f", {output_root / 'rel' / (csv_path.stem + '.rel')}"
                f" (type={coord_type}, cols={col_a}/{col_b})"
            )
            continue

        try:
            geo_path, rel_path, n_nodes, n_edges = _write_geo_rel(
                src_csv=csv_path,
                coord_type=coord_type,
                col_a=col_a,
                col_b=col_b,
                output_root=output_root,
            )
            converted += 1
            print(
                f"[OK] {csv_path} -> {geo_path}, {rel_path} "
                f"(nodes={n_nodes}, edges={n_edges}, type={coord_type})"
            )
        except Exception as e:  # pragma: no cover
            skipped += 1
            print(f"[SKIP] {csv_path} (conversion failed: {e})")

    print("\n[SUMMARY]")
    print(f"  scanned csv files: {total_csv}")
    print(f"  converted:         {converted}")
    print(f"  skipped:           {skipped}")


if __name__ == "__main__":
    main()

