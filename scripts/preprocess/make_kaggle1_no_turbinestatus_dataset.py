from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "data/forecasting/forecasting_wp1_all_shapes_v1/"
    "tfb-kaggle1_final_benchmark_v3_ready.csv"
)
DEFAULT_LONG_OUTPUT = Path(
    "data/forecasting/forecasting_wp1_all_shapes_v1/"
    "tfb-kaggle1_final_benchmark_v3_ready_no_turbinestatus.csv"
)
DEFAULT_WIDE_OUTPUT = Path(
    "data/forecasting/forecasting_wp1_all_shapes_v1/"
    "tfb-kaggle1_final_benchmark_v3_ready_no_turbinestatus_wide.csv"
)
DROP_COL = "type 1:TurbineStatus"


def _validate_long_table(df: pd.DataFrame, input_path: Path) -> None:
    required = {"date", "data", "cols"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} missing required columns: {sorted(missing)}")
    duplicate_count = int(df.duplicated(["date", "cols"]).sum())
    if duplicate_count:
        raise ValueError(f"{input_path} has {duplicate_count} duplicate date/cols rows")


def _write_wide(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    column_order = df["cols"].drop_duplicates().tolist()
    wide = (
        df.pivot(index="date", columns="cols", values="data")
        .reindex(columns=column_order)
        .reset_index()
    )
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.sort_values("date")
    wide["date"] = wide["date"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    wide.to_csv(output_path, index=False)
    return wide


def build_no_turbinestatus_dataset(
    input_path: Path,
    long_output_path: Path,
    wide_output_path: Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not overwrite:
        for output_path in (long_output_path, wide_output_path):
            if output_path.exists():
                raise FileExistsError(
                    f"{output_path} already exists; pass --overwrite to replace it"
                )

    df = pd.read_csv(input_path)
    _validate_long_table(df, input_path)
    if DROP_COL not in set(df["cols"]):
        raise ValueError(f"{input_path} does not contain {DROP_COL!r}")

    filtered = df[df["cols"] != DROP_COL].copy()
    output_parent = long_output_path.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    wide_output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(long_output_path, index=False)
    wide = _write_wide(filtered, wide_output_path)
    return filtered, wide


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create kaggle1 variants without the TurbineStatus column."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--long-output", default=str(DEFAULT_LONG_OUTPUT))
    parser.add_argument("--wide-output", default=str(DEFAULT_WIDE_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    filtered, wide = build_no_turbinestatus_dataset(
        Path(args.input),
        Path(args.long_output),
        Path(args.wide_output),
        overwrite=args.overwrite,
    )
    print(f"wrote {args.long_output}")
    print(f"wrote {args.wide_output}")
    print(f"long_shape={filtered.shape}")
    print(f"wide_shape={wide.shape}")
    print(f"dropped_col={DROP_COL}")
    print(f"remaining_cols={filtered['cols'].nunique()}")
    print("remaining_col_names:")
    for idx, col in enumerate(filtered["cols"].drop_duplicates()):
        print(f"  {idx}: {col}")
    numeric = wide.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
    print("nan_count=", int(numeric.isna().sum().sum()))
    print("max_abs=", float(numeric.abs().max().max()))


if __name__ == "__main__":
    main()
