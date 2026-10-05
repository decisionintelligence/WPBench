from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "data/forecasting/forecasting_wp1_all_shapes_v1/"
    "tfb-kaggle1_final_benchmark_v3_ready.csv"
)
DEFAULT_OUTPUT = Path(
    "data/forecasting/forecasting_wp1_all_shapes_v1/"
    "tfb-kaggle1_final_benchmark_v3_ready_wide.csv"
)


def build_kaggle1_wide(input_path: Path, output_path: Path, overwrite: bool = False) -> pd.DataFrame:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    df = pd.read_csv(input_path)
    required = {"date", "data", "cols"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} missing required columns: {sorted(missing)}")

    duplicate_count = int(df.duplicated(["date", "cols"]).sum())
    if duplicate_count:
        raise ValueError(f"{input_path} has {duplicate_count} duplicate date/cols rows")

    column_order = df["cols"].drop_duplicates().tolist()
    wide = (
        df.pivot(index="date", columns="cols", values="data")
        .reindex(columns=column_order)
        .reset_index()
    )
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.sort_values("date")
    wide["date"] = wide["date"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False)
    return wide


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the kaggle1 benchmark long table into a non-overwriting wide table."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    wide = build_kaggle1_wide(input_path, output_path, overwrite=args.overwrite)
    print(f"wrote {output_path}")
    print(f"shape={wide.shape}")
    print("columns:")
    for idx, col in enumerate(wide.columns):
        print(f"  {idx}: {col}")
    numeric = wide.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
    print("nan_count=", int(numeric.isna().sum().sum()))
    print("max_abs=", float(numeric.abs().max().max()))


if __name__ == "__main__":
    main()
