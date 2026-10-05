# Build `.geo` and `.rel` From Coordinate CSVs

This utility scans dataset folders, detects coordinate columns, and generates:

- `.geo` files with columns: `node_id,Lat,Lng`
- `.rel` files with columns: `origin_id,destination_id,cost`

The generated `.rel` uses directed edges for all node pairs except self-loops.

## Script

- `scripts/build_geo_rel_from_coords.py`

## Default scanned folders

- `dataset/Kelmarsh`
- `dataset/Penmanshiel`
- `dataset/wind_Spatio-Temporal_Dataset2`
- `dataset/sdwpf`

## Coordinate detection

- Lat/Lon: `lat`/`latitude` + `lon`/`lng`/`longitude`
- XY fallback: `x` + `y`

For XY datasets, output still uses `Lat`/`Lng` columns for compatibility with existing loader code.

## Usage

Dry run:

```bash
python scripts/build_geo_rel_from_coords.py --dry-run
```

Generate sidecar outputs next to source CSVs (`<source_dir>/geo/*.geo`, `<source_dir>/rel/*.rel`):

```bash
python scripts/build_geo_rel_from_coords.py
```

Generate central outputs under `dataset/forecasting`:

```bash
python scripts/build_geo_rel_from_coords.py --output-mode central --central-output-root dataset/forecasting
```

## Notes

- Lat/Lon datasets use haversine distance in meters.
- XY datasets use Euclidean distance in source unit.
- Node IDs are re-indexed to contiguous `0..N-1`.

