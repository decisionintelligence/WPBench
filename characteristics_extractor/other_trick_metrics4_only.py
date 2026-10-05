import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy, norm
from tqdm import tqdm

warnings.filterwarnings("ignore")


class Metrics4Extractor:
    """
    Reproduce exactly the 4 indicators used in other_trick_add.py:
    - Transition  <- SB_TransitionMatrix_3ac_sumdiagcov (catch22)
    - Shifting    <- abs(DN_OutlierInclude_p_001_mdrmd) (catch22)
    - Short_term_jsd <- JSD with window=30
    - Long_term_jsd  <- JSD with window=336
    """

    def __init__(self, enable_r: bool = True):
        self.enable_r = enable_r
        self._r_ready = False
        self._calculate_transition_shifting = None
        if self.enable_r:
            self._setup_r_environment()

    @staticmethod
    def read_data(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        data = pd.read_csv(path)
        label_exists = "label" in data["cols"].values

        all_points = data.shape[0]
        columns = data.columns

        if columns[0] == "date":
            n_points = data.iloc[:, 2].value_counts().max()
        else:
            n_points = data.iloc[:, 1].value_counts().max()

        is_univariate = n_points == all_points
        n_cols = all_points // n_points
        df = pd.DataFrame()
        cols_name = data["cols"].unique()

        if columns[0] == "date" and not is_univariate:
            df["date"] = data.iloc[:n_points, 0]
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        elif columns[0] != "date" and not is_univariate:
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

        elif columns[0] == "date" and is_univariate:
            df["date"] = data.iloc[:, 0]
            df[cols_name[0]] = data.iloc[:, 1]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        else:
            df[cols_name[0]] = data.iloc[:, 0]

        if label_exists:
            last_col_name = df.columns[-1]
            df.rename(columns={last_col_name: "label"}, inplace=True)
            df = df.drop(columns="label")

        if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
            df = df.iloc[:nrows, :]

        return df

    def _setup_r_environment(self) -> None:
        try:
            from rpy2 import robjects
            from rpy2.robjects import pandas2ri

            pandas2ri.activate()
            r_script = """
            library(tidyverse)
            library(Rcatch22)
            library(forecast)

            calculate_transition_shifting <- function(dir_name, data) {
                out <- tibble(
                    file_name = dir_name,
                    SB_TransitionMatrix_3ac_sumdiagcov = NA_real_,
                    DN_OutlierInclude_p_001_mdrmd = NA_real_
                )

                tryCatch(
                    expr = {
                        ts <- forecast:::msts(data, seasonal.periods = 1)
                        catch_features <- catch22_all(ts)

                        idx_t <- which(catch_features$names == "SB_TransitionMatrix_3ac_sumdiagcov")
                        idx_s <- which(catch_features$names == "DN_OutlierInclude_p_001_mdrmd")

                        if (length(idx_t) > 0) {
                            out$SB_TransitionMatrix_3ac_sumdiagcov <- catch_features$values[idx_t[1]]
                        }
                        if (length(idx_s) > 0) {
                            out$DN_OutlierInclude_p_001_mdrmd <- catch_features$values[idx_s[1]]
                        }
                    },
                    error = function(e) {
                        out$SB_TransitionMatrix_3ac_sumdiagcov <- NA_real_
                        out$DN_OutlierInclude_p_001_mdrmd <- NA_real_
                    }
                )

                return(out)
            }
            """
            robjects.r(r_script)
            self._calculate_transition_shifting = robjects.globalenv["calculate_transition_shifting"]
            self._FloatVector = robjects.FloatVector
            self._pandas2ri = pandas2ri
            self._r_ready = True
        except Exception as exc:
            self._r_ready = False
            self._setup_error = str(exc)

    @staticmethod
    def calculate_jsd_for_window(data: np.ndarray, window_size: int) -> float:
        jsd_list = []
        num_windows = len(data) // window_size

        for i in range(num_windows):
            window_data = data[i * window_size : (i + 1) * window_size]
            hist, bin_edges = np.histogram(window_data, bins="stone", density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            mu = np.mean(window_data)
            sigma = np.std(window_data)

            if sigma == 0:
                jsd_list.append(0)
                continue

            pdf = norm.pdf(bin_centers, mu, sigma)
            jsd = Metrics4Extractor.js_divergence(hist, pdf)
            jsd_list.append(jsd)

        return float(np.mean(jsd_list)) if jsd_list else np.nan

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        kl_p_m = entropy(p, m)
        kl_q_m = entropy(q, m)
        return float(0.5 * (kl_p_m + kl_q_m))

    @staticmethod
    def calculate_jsd_multivariate(df: pd.DataFrame, window_size: int) -> List[float]:
        return [Metrics4Extractor.calculate_jsd_for_window(df[col].values, window_size) for col in df.columns]

    def calculate_jsd(self, filename: str) -> pd.DataFrame:
        df = self.read_data(path=filename)
        df = self.select_first_target_per_turbid(df)
        short_term_jsd = self.calculate_jsd_multivariate(df, 30)
        long_term_jsd = self.calculate_jsd_multivariate(df, 336)
        return pd.DataFrame({"short_term_jsd": short_term_jsd, "long_term_jsd": long_term_jsd})

    @staticmethod
    def select_first_target_per_turbid(df: pd.DataFrame) -> pd.DataFrame:
        # Match aligned extractor logic: per TurbID, keep only the first variable column.
        selected = []
        seen = set()
        for col in df.columns:
            tid = str(col).split(":")[0]
            if tid not in seen:
                seen.add(tid)
                selected.append(col)
        return df[selected].copy()

    def process_file(self, file_path: str) -> pd.DataFrame:
        data = self.read_data(file_path)
        data = self.select_first_target_per_turbid(data)
        col_names = data.columns.tolist()
        turbids = [str(c).split(":")[0] for c in col_names]

        jsd_result = self.calculate_jsd(file_path)

        r_rows = []
        if self.enable_r and not self._r_ready:
            raise RuntimeError(
                "R feature backend is unavailable. Install rpy2 + R packages (tidyverse/Rcatch22/forecast), "
                f"or run with --skip-r. setup_error={getattr(self, '_setup_error', 'unknown')}"
            )

        if self.enable_r and self._r_ready:
            for i, col in enumerate(col_names):
                col_data = data.iloc[:, i].dropna().tolist()
                r_data = self._FloatVector(col_data)
                r_feat = self._calculate_transition_shifting(turbids[i], r_data)
                r_df = self._pandas2ri.rpy2py(r_feat)
                r_rows.append(r_df)

            transition_df = pd.concat(r_rows, ignore_index=True)
        else:
            transition_df = pd.DataFrame(
                {
                    "file_name": turbids,
                    "SB_TransitionMatrix_3ac_sumdiagcov": [np.nan] * len(col_names),
                    "DN_OutlierInclude_p_001_mdrmd": [np.nan] * len(col_names),
                }
            )

        result = pd.concat([jsd_result, transition_df], axis=1)
        result["DN_OutlierInclude_p_001_mdrmd"] = result["DN_OutlierInclude_p_001_mdrmd"].abs()
        result.insert(0, "TurbID", turbids)
        result.insert(0, "Series", col_names)
        return result

    @staticmethod
    def to_tfb_metric_schema(df: pd.DataFrame) -> pd.DataFrame:
        out = df.loc[
            :,
            [
                "Series",
                "TurbID",
                "SB_TransitionMatrix_3ac_sumdiagcov",
                "DN_OutlierInclude_p_001_mdrmd",
                "short_term_jsd",
                "long_term_jsd",
            ],
        ].copy()
        out.columns = [
            "Series",
            "TurbID",
            "Transition",
            "Shifting",
            "Short_term_jsd",
            "Long_term_jsd",
        ]
        return out


class Metrics4Processor:
    def __init__(self, output_dir: str, enable_r: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = Metrics4Extractor(enable_r=enable_r)

    def process_path(self, file_path: str, pattern: str = "*.csv", verbose: bool = False, print_every: int = 20) -> None:
        p = Path(file_path)
        if p.is_file() and p.suffix.lower() == ".csv":
            self._process_single_file(p)
            return

        if p.is_dir():
            files = sorted(p.glob(pattern))
            if not files:
                print(f"[WARN] no files matched pattern '{pattern}' under: {p}")
                return
            self.process_files([f for f in files if f.is_file()], verbose=verbose, print_every=print_every)
            return

        raise ValueError(f"Invalid path: {file_path}")

    def process_files(self, files: List[Path], verbose: bool = False, print_every: int = 20) -> None:
        detail_dir = self.output_dir / "dataset_details"
        detail_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: List[Dict[str, Any]] = []
        all_detail_rows: List[Dict[str, Any]] = []

        for idx, file_path in enumerate(tqdm(files, desc="Metrics4 Benchmarking"), start=1):
            if verbose:
                tqdm.write(f"[INFO] start dataset: {file_path.stem}")
            try:
                final_df = self._process_single_file(file_path)
            except Exception as exc:
                print(f"[WARN] skip failed dataset: {file_path} | {exc}")
                continue

            base = file_path.stem
            detail_rows = final_df.copy()
            detail_rows.insert(0, "Dataset", base)
            all_detail_rows.extend(detail_rows.to_dict(orient="records"))

            summary_rows.append(
                {
                    "Dataset": base,
                    "Transition": float(pd.to_numeric(final_df["Transition"], errors="coerce").mean()),
                    "Shifting": float(pd.to_numeric(final_df["Shifting"], errors="coerce").mean()),
                    "Short_term_jsd": float(pd.to_numeric(final_df["Short_term_jsd"], errors="coerce").mean()),
                    "Long_term_jsd": float(pd.to_numeric(final_df["Long_term_jsd"], errors="coerce").mean()),
                    "NumTurbines": int(len(final_df)),
                }
            )

            detail_file = detail_dir / f"{base}_detail.csv"
            detail_rows.to_csv(detail_file, index=False)
            if verbose and (idx == 1 or idx % max(1, int(print_every)) == 0 or idx == len(files)):
                tqdm.write(
                    f"[INFO] done dataset: {base} | turbines={len(final_df)} | "
                    f"mean_transition={pd.to_numeric(final_df['Transition'], errors='coerce').mean():.4f}"
                )

        pd.DataFrame(summary_rows).to_csv(self.output_dir / "Final_Metrics4_Report.csv", index=False)
        pd.DataFrame(all_detail_rows).to_csv(self.output_dir / "Dataset_Metrics4_Detail_Report.csv", index=False)
        print(f"[INFO] summary saved: {self.output_dir / 'Final_Metrics4_Report.csv'}")
        print(f"[INFO] details saved: {self.output_dir / 'Dataset_Metrics4_Detail_Report.csv'}")
        print(f"[INFO] per-dataset details dir: {detail_dir}")

    def _process_single_file(self, file_path: Path) -> pd.DataFrame:
        print(f"[INFO] processing file: {file_path}")
        base = file_path.stem

        raw = self.extractor.process_file(str(file_path))
        final = self.extractor.to_tfb_metric_schema(raw)

        raw_file = self.output_dir / f"All_metrics4_{base}.csv"
        final_file = self.output_dir / f"TFB_metrics4_{base}.csv"
        mean_file = self.output_dir / f"mean_TFB_metrics4_{base}.csv"

        raw.to_csv(raw_file, index=False)
        final.to_csv(final_file, index=False)

        mean_numeric = final[["Transition", "Shifting", "Short_term_jsd", "Long_term_jsd"]].mean(numeric_only=True)
        mean_df = pd.DataFrame([mean_numeric.to_dict()])
        mean_df.to_csv(mean_file, index=False)

        print(f"[INFO] saved: {raw_file}")
        print(f"[INFO] saved: {final_file}")
        print(f"[INFO] saved: {mean_file}")
        return final


def _run_self_test(output_dir: Path) -> None:
    src = output_dir / "_selftest_src"
    src.mkdir(parents=True, exist_ok=True)

    n = 360
    t = pd.date_range("2020-01-01", periods=n, freq="10min")
    rows = []
    for name in ["type 1:Power (kW)", "type 2:Power (kW)"]:
        if "type 1" in name:
            data = np.sin(np.arange(n) / 12.0)
        else:
            data = 0.7 * np.sin(np.arange(n) / 9.0) + 0.1 * np.random.randn(n)
        rows.append(pd.DataFrame({"date": t, "data": data, "cols": name}))

    test_csv = src / "tfb-selftest.csv"
    pd.concat(rows, ignore_index=True).to_csv(test_csv, index=False)

    processor = Metrics4Processor(output_dir=str(output_dir), enable_r=False)
    processor.process_path(str(test_csv))

    assert (output_dir / "All_metrics4_tfb-selftest.csv").exists()
    assert (output_dir / "TFB_metrics4_tfb-selftest.csv").exists()
    assert (output_dir / "mean_TFB_metrics4_tfb-selftest.csv").exists()
    print("[INFO] self-test passed (JSD path validated, R metrics skipped)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Transition/Shifting/Short_term_jsd/Long_term_jsd only.")
    parser.add_argument("--input", required=False, default=None, help="CSV file or directory")
    parser.add_argument("--src", required=False, default=None, help="Source directory (aligned-style alias of --input)")
    parser.add_argument("--out", required=False, default="./characteristics_metrics4", help="Output directory")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern when --input is a directory")
    parser.add_argument("--skip-r", action="store_true", help="Skip Transition/Shifting R backend and output NaN for them")
    parser.add_argument("--verbose", action="store_true", help="Print progress details while running")
    parser.add_argument("--print-every", type=int, default=20, help="When --verbose is set, print every N datasets")
    parser.add_argument("--self-test", action="store_true", help="Run built-in self test")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        _run_self_test(out_path)
        return

    input_path = args.input or args.src or "/home/dag-test-zhu/characteristics_extractor/DemoDatasets/Exchange.csv"
    processor = Metrics4Processor(output_dir=str(out_path), enable_r=not args.skip_r)
    processor.process_path(input_path, pattern=args.pattern, verbose=args.verbose, print_every=args.print_every)


if __name__ == "__main__":
    main()

