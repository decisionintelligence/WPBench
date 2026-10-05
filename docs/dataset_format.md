# Dataset Format

WPBench forecasting datasets are CSV files loaded from `data/forecasting/`.

## Wide Format

Required first column:

- `date`: timestamp parseable by pandas.

Remaining columns are numeric target/covariate series. Example:

```csv
date,type 1:Power,type 2:WindSpeed
2020-01-01 00:00:00,12.3,5.6
```

## Long Format

Required columns:

- `date`: timestamp.
- `cols`: variable/channel name.
- `data`: numeric value.

Example:

```csv
date,cols,data
2020-01-01 00:00:00,type 1:Power,12.3
```

## Metadata

`FORECAST_META.csv` is stored at both `data/forecasting/FORECAST_META.csv` for the loader and `data/metadata/FORECAST_META.csv` for documentation. If a CSV is missing from metadata, the original loader can append a basic metadata row.

## Graph Data

Small STWave graph arrays are in `data/graphs/tem_graph/`. Relation/geo sidecars can be generated with `scripts/preprocess/build_geo_rel_from_coords.py` when coordinate CSVs are available.
