from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = ARTIFACT_ROOT / "data" / "forecasting" / "forecasting_wp1_all_shapes_v1"

MAELSTROM_FILES = [
    "tfb-maelstrom_turbine_2_univariate.csv",
    "tfb-maelstrom_turbine_3_univariate.csv",
    "tfb-maelstrom_turbine_4_univariate.csv",
]

CHALMERS_FILE = (
    "tfb-chalmers_merged_processed_data_fixed_combined__1T1V_reduced_1min_1T_1V_power.csv"
)


def normalize_maelstrom_cols(
    dataset_dir: Path,
    *,
    apply: bool,
    backup: bool,
) -> None:
    """Normalize single-turbine Maelstrom columns to the shared type 1 label."""
    for name in MAELSTROM_FILES:
        path = dataset_dir / name
        df = pd.read_csv(path)
        if "cols" not in df.columns:
            raise ValueError(f"{path} does not contain a 'cols' column")

        before = df["cols"].astype(str)
        after = before.str.replace(r"^type\s+\d+:", "type 1:", regex=True)
        changed = int((before != after).sum())
        before_unique = sorted(before.unique())
        after_unique = sorted(after.unique())

        print(f"[MAELSTROM] file={path}")
        print(f"  before_unique={before_unique}")
        print(f"  after_unique={after_unique}")
        print(f"  changed_rows={changed}")

        if not apply:
            continue

        if changed == 0:
            print("  write=skipped_no_change")
            continue

        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak_before_type1_fix")
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
                print(f"  backup={backup_path}")
            else:
                print(f"  backup=exists:{backup_path}")

        df["cols"] = after
        df.to_csv(path, index=False)
        print("  write=done")


def diagnose_chalmers_timestamps(
    dataset_dir: Path,
    *,
    expected_freq: str,
    max_examples: int,
    output_bad_gaps: Path | None,
) -> None:
    """Print timestamp regularity diagnostics for the Chalmers dataset."""
    path = dataset_dir / CHALMERS_FILE
    df = pd.read_csv(path, usecols=["date"])
    ts = pd.to_datetime(df["date"], errors="coerce")
    non_null = ts.dropna()
    expected_delta = pd.Timedelta(expected_freq)
    deltas = ts.diff().dropna()
    bad = deltas[deltas != expected_delta]

    print(f"[CHALMERS] file={path}")
    print(f"  rows={len(ts)}")
    print(f"  nat_count={int(ts.isna().sum())}")
    print(f"  duplicate_count={int(ts.duplicated().sum())}")
    print(f"  monotonic_increasing={bool(ts.is_monotonic_increasing)}")
    print(f"  first={ts.iloc[0] if len(ts) else ''}")
    print(f"  last={ts.iloc[-1] if len(ts) else ''}")
    print(f"  infer_freq={pd.infer_freq(non_null) if len(non_null) >= 3 else None}")
    print(f"  expected_delta={expected_delta}")
    print(f"  unique_delta_count={deltas.nunique()}")
    print("  top_deltas:")
    print(deltas.value_counts().head(20).to_string())
    print(f"  bad_delta_count={len(bad)}")

    if len(bad):
        rows = []
        print("  first_bad_gaps:")
        for idx, delta in bad.head(max_examples).items():
            prev_ts = ts.iloc[idx - 1]
            curr_ts = ts.iloc[idx]
            missing_steps = int(delta / expected_delta) - 1 if delta > expected_delta else 0
            rows.append(
                {
                    "row_index": idx,
                    "previous_timestamp": prev_ts,
                    "current_timestamp": curr_ts,
                    "delta": delta,
                    "missing_expected_steps": missing_steps,
                }
            )
            print(
                "    "
                f"row={idx} previous={prev_ts} current={curr_ts} "
                f"delta={delta} missing_expected_steps={missing_steps}"
            )

        if output_bad_gaps is not None:
            output_bad_gaps.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(output_bad_gaps, index=False)
            print(f"  wrote_bad_gap_examples={output_bad_gaps}")


def rewrite_chalmers_timestamps(
    path: Path,
    *,
    expected_freq: str,
    apply: bool,
    backup: bool,
) -> None:
    """Rewrite Chalmers date values as a contiguous timeline, preserving row order."""
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} does not contain a 'date' column")

    ts = pd.to_datetime(df["date"], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{path} contains invalid timestamps")
    if len(ts) == 0:
        print(f"[CHALMERS_FIX] file={path} rows=0 write=skipped_empty")
        return

    new_ts = pd.date_range(start=ts.iloc[0], periods=len(ts), freq=expected_freq)
    changed = int((ts.reset_index(drop=True) != pd.Series(new_ts)).sum())
    print(f"[CHALMERS_FIX] file={path}")
    print(f"  start={new_ts[0]}")
    print(f"  end={new_ts[-1]}")
    print(f"  rows={len(ts)}")
    print(f"  changed_rows={changed}")

    if not apply:
        print("  write=skipped_dry_run")
        return

    if changed == 0:
        print("  write=skipped_no_change")
        return

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak_before_contiguous_time_fix")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"  backup={backup_path}")
        else:
            print(f"  backup=exists:{backup_path}")

    df["date"] = new_ts.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(path, index=False)
    print("  write=done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixes for WP1 all-shapes dataset issues. By default this "
            "runs in dry-run mode and does not modify CSV files."
        )
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite Maelstrom turbine 2-4 cols values.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak_before_type1_fix files when --apply is used.",
    )
    parser.add_argument(
        "--fix-chalmers-contiguous",
        action="store_true",
        help=(
            "Rewrite the Chalmers date column as a contiguous expected-freq timeline. "
            "Only writes when --apply is also set."
        ),
    )
    parser.add_argument("--expected-freq", default="1min")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument(
        "--output-bad-gaps",
        default=str(ARTIFACT_ROOT / "results" / "aggregated" / "wp1_all_shapes_dataset_issue_diagnostics" / "chalmers_bad_gaps.csv"),
        help="Where to write Chalmers bad-gap examples. Use empty string to skip.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_bad_gaps = Path(args.output_bad_gaps) if args.output_bad_gaps else None

    normalize_maelstrom_cols(
        dataset_dir,
        apply=args.apply,
        backup=not args.no_backup,
    )
    diagnose_chalmers_timestamps(
        dataset_dir,
        expected_freq=args.expected_freq,
        max_examples=args.max_examples,
        output_bad_gaps=output_bad_gaps,
    )
    if args.fix_chalmers_contiguous:
        rewrite_chalmers_timestamps(
            dataset_dir / CHALMERS_FILE,
            expected_freq=args.expected_freq,
            apply=args.apply,
            backup=not args.no_backup,
        )

    if not args.apply:
        print("[DRY_RUN] CSV files were not modified. Re-run with --apply to write fixes.")


if __name__ == "__main__":
    main()
