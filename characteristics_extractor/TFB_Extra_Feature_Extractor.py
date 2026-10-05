import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import eigvals
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

warnings.filterwarnings("ignore")


class TFBExtraFeatureExtractor:
    """
    Extract only the extra metrics introduced beyond the aligned season/trend logic.

    Scope of this extractor:
    - CPCD_Residual
    - Var_Corr
    - Var_TGV_72
    - Var_GSD_72
    - Spatial_Corr
    - Spatial_TGV_72
    - Spatial_GSD_72

    The aligned macro metrics (Seasonality / Trend / Stationarity) should be computed
    by `Characteristics_Extractor_tfb_aligned.py` and kept in a separate output tree.
    """

    def __init__(self, var_windows: Optional[List[int]] = None, spatial_windows: Optional[List[int]] = None):
        self.VAR_WS = var_windows or [72, 576]
        self.SPATIAL_WS = spatial_windows or [72, 576]
        self.cpcd_mapping = {
            "gefcom2012": lambda df, cols: (df[cols[0]].values, df[cols[3]].values),
            "gefcom2014": self._calc_gefcom2014_ws,
            "kaggle1": lambda df, cols: (df[cols[0]].values, df[cols[-1]].values),
            "scada_fault": lambda df, cols: (df[cols[0]].values, df[cols[2]].values),
        }

    def _calc_gefcom2014_ws(self, df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        p = pd.to_numeric(df[cols[0]], errors="coerce").values
        u100 = pd.to_numeric(df[cols[3]], errors="coerce").values
        v100 = pd.to_numeric(df[cols[4]], errors="coerce").values
        return p, np.sqrt(u100 ** 2 + v100 ** 2)

    def read_tfb_data(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        n_points = data.iloc[:, 2].value_counts().max()
        cols_name = data["cols"].unique()
        df = pd.DataFrame()
        df["date"] = data.iloc[:n_points, 0]
        for j, name in enumerate(cols_name):
            df[name] = data.iloc[j * n_points : (j + 1) * n_points, 1].values
        return df.set_index("date")

    def calculate_cpcd(self, p: np.ndarray, v: np.ndarray) -> float:
        p_num = pd.to_numeric(pd.Series(p), errors="coerce").values.astype(float)
        v_num = pd.to_numeric(pd.Series(v), errors="coerce").values.astype(float)
        mask = ~np.isnan(p_num) & ~np.isnan(v_num)
        if not np.any(mask):
            return np.nan
        p_ok = p_num[mask]
        v3 = v_num[mask] ** 3
        denom = np.sum(v3 ** 2)
        c = np.sum(p_ok * v3) / denom if denom != 0 else 0.0
        return float(np.mean((p_ok - c * v3) ** 2) / (np.var(p_ok) + 1e-6))

    def _group_turbines(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        turbines: Dict[str, List[str]] = {}
        for c in df.columns:
            tid = str(c).split(":")[0]
            turbines.setdefault(tid, []).append(c)
        return turbines

    def _to_numeric_matrix(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        return np.column_stack([pd.to_numeric(df[c], errors="coerce").values for c in cols]).astype(float)

    def _safe_abs_target_corr(self, matrix: np.ndarray) -> float:
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            return np.nan
        corr = np.corrcoef(matrix.T)
        if corr.ndim != 2 or corr.shape[0] < 2:
            return np.nan
        vals = np.abs(corr[0, 1:])
        vals = vals[~np.isnan(vals)]
        return float(np.mean(vals)) if len(vals) else np.nan

    def _safe_abs_upper_corr(self, matrix: np.ndarray) -> float:
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            return np.nan
        corr = np.corrcoef(matrix.T)
        if corr.ndim != 2 or corr.shape[0] < 2:
            return np.nan
        tri = np.abs(corr[np.triu_indices(corr.shape[0], k=1)])
        tri = tri[~np.isnan(tri)]
        return float(np.mean(tri)) if len(tri) else np.nan

    def _compute_dynamics(self, matrix: np.ndarray, window_size: int, mode: str = "variable") -> Tuple[float, float]:
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            return np.nan, np.nan

        num_windows = len(data) // window_size
        if num_windows < 2:
            return np.nan, np.nan

        tgvs: List[float] = []
        gsds: List[float] = []
        prev_adj = None
        prev_eigs = None
        dim = data.shape[1]

        for i in range(num_windows):
            window = np.nan_to_num(data[i * window_size : (i + 1) * window_size], nan=0.0)
            if mode == "variable":
                target = window[:, 0:1]
                covs = window[:, 1:]
                if covs.shape[1] == 0:
                    return np.nan, np.nan
                current_feat = np.nan_to_num(cosine_similarity(target.T, covs.T).flatten(), nan=0.0)
                adj = np.zeros((dim, dim), dtype=float)
                adj[0, 1:] = current_feat
                adj[1:, 0] = current_feat
            else:
                adj = np.nan_to_num(cosine_similarity(window.T), nan=0.0)
                np.fill_diagonal(adj, 1.0)
                current_feat = adj

            l_unnorm = np.diag(adj.sum(axis=1)) - adj
            try:
                eigs = np.sort(np.real(eigvals(l_unnorm)))
            except Exception:
                continue

            if prev_adj is not None:
                if mode == "variable":
                    denom_tgv = np.sqrt(dim - 1) if dim > 1 else np.nan
                    denom_gsd = dim
                else:
                    denom_tgv = dim
                    denom_gsd = dim * np.sqrt(dim)

                if denom_tgv and denom_gsd:
                    tgvs.append(float(np.linalg.norm(current_feat - prev_adj) / denom_tgv))
                    gsds.append(float(np.linalg.norm(eigs - prev_eigs, 2) / denom_gsd))

            prev_adj = current_feat
            prev_eigs = eigs

        return (
            float(np.nanmean(tgvs)) if len(tgvs) else np.nan,
            float(np.nanmean(gsds)) if len(gsds) else np.nan,
        )

    def _pick_cpcd_pair(self, prefix: str, df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        cpcd_key = next((k for k in self.cpcd_mapping.keys() if k in prefix.lower()), None)
        if cpcd_key:
            return self.cpcd_mapping[cpcd_key](df, cols)
        if len(cols) < 2:
            raise ValueError("Need at least two columns to compute fallback CPCD")
        p = pd.to_numeric(df[cols[0]], errors="coerce").values
        v = pd.to_numeric(df[cols[1]], errors="coerce").values
        return p, v

    def process(self, files: List[str], out_dir: str, verbose: bool = False, print_every: int = 20) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        detail_dir = out_path / "dataset_extra_details"
        detail_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: List[Dict[str, Any]] = []
        detail_rows_all: List[Dict[str, Any]] = []

        for f_path in tqdm(files, desc="TFB Extra Feature Benchmarking"):
            prefix = os.path.splitext(os.path.basename(f_path))[0]
            if verbose:
                tqdm.write(f"[INFO] start dataset: {prefix}")
            try:
                df = self.read_tfb_data(f_path)
            except Exception as exc:
                print(f"[WARN] skip unreadable file: {f_path} | {exc}")
                continue

            turbines = self._group_turbines(df)
            dataset_detail_rows: List[Dict[str, Any]] = []
            cpcds: List[float] = []
            var_corrs: List[float] = []
            v_tgvs: List[float] = []
            v_gsds: List[float] = []

            for idx, (tid, cols) in enumerate(turbines.items(), start=1):
                power_col = cols[0]
                one_row: Dict[str, Any] = {
                    "Dataset": prefix,
                    "TurbID": tid,
                    "TargetColumn": power_col,
                    "NumVariables": int(len(cols)),
                    "CPCD_Residual": np.nan,
                    "Var_Corr": np.nan,
                    "Var_TGV_72": np.nan,
                    "Var_GSD_72": np.nan,
                }

                if len(cols) > 1:
                    matrix = self._to_numeric_matrix(df, cols)
                    one_row["Var_Corr"] = self._safe_abs_target_corr(matrix)
                    one_row["Var_TGV_72"], one_row["Var_GSD_72"] = self._compute_dynamics(
                        matrix, self.VAR_WS[0], mode="variable"
                    )
                    try:
                        p_c, w_c = self._pick_cpcd_pair(prefix, df, cols)
                        one_row["CPCD_Residual"] = self.calculate_cpcd(p_c, w_c)
                    except Exception:
                        pass

                dataset_detail_rows.append(one_row)
                detail_rows_all.append(one_row)

                for src_list, key in [
                    (cpcds, "CPCD_Residual"),
                    (var_corrs, "Var_Corr"),
                    (v_tgvs, "Var_TGV_72"),
                    (v_gsds, "Var_GSD_72"),
                ]:
                    value = one_row[key]
                    if not pd.isna(value):
                        src_list.append(float(value))

                if verbose and (idx == 1 or idx % max(1, int(print_every)) == 0 or idx == len(turbines)):
                    tqdm.write(
                        f"[INFO] {prefix} | turb {idx}/{len(turbines)} | "
                        f"TurbID={tid} | CPCD={one_row['CPCD_Residual']:.4f} "
                        f"VarCorr={one_row['Var_Corr']:.4f}"
                    )

            dataset_row: Dict[str, Any] = {
                "Dataset": prefix,
                "CPCD_Residual": float(np.nanmean(cpcds)) if cpcds else np.nan,
                "Var_Corr": float(np.nanmean(var_corrs)) if var_corrs else np.nan,
                "Var_TGV_72": float(np.nanmean(v_tgvs)) if v_tgvs else np.nan,
                "Var_GSD_72": float(np.nanmean(v_gsds)) if v_gsds else np.nan,
                "Spatial_Corr": np.nan,
                "Spatial_TGV_72": np.nan,
                "Spatial_GSD_72": np.nan,
                "NumTurbines": int(len(dataset_detail_rows)),
            }

            power_cols = [cols[0] for cols in turbines.values()]
            if len(power_cols) > 1:
                p_data = self._to_numeric_matrix(df, power_cols)
                dataset_row["Spatial_Corr"] = self._safe_abs_upper_corr(p_data)
                dataset_row["Spatial_TGV_72"], dataset_row["Spatial_GSD_72"] = self._compute_dynamics(
                    p_data, self.SPATIAL_WS[0], mode="spatial"
                )

            summary_rows.append(dataset_row)

            if dataset_detail_rows:
                pd.DataFrame(dataset_detail_rows).to_csv(detail_dir / f"{prefix}_extra_detail.csv", index=False)
                if verbose:
                    tqdm.write(
                        f"[INFO] done dataset: {prefix} | turbines={len(dataset_detail_rows)} | "
                        f"mean_CPCD={dataset_row['CPCD_Residual']:.4f} "
                        f"mean_VarCorr={dataset_row['Var_Corr']:.4f}"
                    )

        pd.DataFrame(summary_rows).to_csv(out_path / "Final_Extra_Feature_Report.csv", index=False)
        pd.DataFrame(detail_rows_all).to_csv(out_path / "Dataset_Extra_Detail_Report.csv", index=False)
        print(f"[INFO] extra summary saved: {out_path / 'Final_Extra_Feature_Report.csv'}")
        print(f"[INFO] extra details saved: {out_path / 'Dataset_Extra_Detail_Report.csv'}")
        print(f"[INFO] per-dataset extra details dir: {detail_dir}")


def _run_self_test() -> None:
    root = Path("./_tfb_extra_extractor_self_test")
    src = root / "src"
    out = root / "out"
    src.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    n = 300
    t = pd.date_range("2020-01-01", periods=n, freq="10min")
    rows = []

    for name in [
        "type 1:Power (kW)",
        "type 1:Wind speed (m/s)",
        "type 2:Power (kW)",
        "type 2:Wind speed (m/s)",
    ]:
        if "Power" in name:
            offset = 1.0 if "type 2" in name else 0.0
            data = np.sin(np.arange(n) / 24.0) + offset
        else:
            offset = 0.2 if "type 2" in name else 0.0
            data = np.cos(np.arange(n) / 24.0) + 5.0 + offset
        rows.append(pd.DataFrame({"date": t, "data": data, "cols": name}))

    tfb_path = src / "tfb-selftest.csv"
    pd.concat(rows, ignore_index=True).to_csv(tfb_path, index=False)

    extractor = TFBExtraFeatureExtractor()
    extractor.process([str(tfb_path)], str(out), verbose=True, print_every=1)

    summary_path = out / "Final_Extra_Feature_Report.csv"
    detail_path = out / "Dataset_Extra_Detail_Report.csv"
    per_dataset_path = out / "dataset_extra_details" / "tfb-selftest_extra_detail.csv"
    assert summary_path.exists()
    assert detail_path.exists()
    assert per_dataset_path.exists()

    summary_df = pd.read_csv(summary_path)
    detail_df = pd.read_csv(detail_path)
    assert not summary_df.empty
    assert not detail_df.empty
    expected_cols = {
        "Dataset",
        "CPCD_Residual",
        "Var_Corr",
        "Var_TGV_72",
        "Var_GSD_72",
        "Spatial_Corr",
        "Spatial_TGV_72",
        "Spatial_GSD_72",
        "NumTurbines",
    }
    assert expected_cols.issubset(set(summary_df.columns))
    print("[INFO] self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract extra TFB metrics that are outside the aligned season/trend/stationarity logic."
    )
    parser.add_argument("--src", default="/home/dag-test-zhu/dataset/forecasting_wp1", help="Source directory")
    parser.add_argument(
        "--out",
        default="/home/dag-test-zhu/dataset/wind_final_stats_extra_metrics",
        help="Output directory",
    )
    parser.add_argument("--pattern", default="tfb-*.csv", help="File glob pattern under --src")
    parser.add_argument("--verbose", action="store_true", help="Print progress details while running")
    parser.add_argument("--print-every", type=int, default=20, help="When --verbose is set, print every N turbines")
    parser.add_argument("--self-test", action="store_true", help="Run built-in self test")
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    src_path = Path(args.src)
    files = sorted(str(p) for p in src_path.glob(args.pattern) if p.is_file())
    if not files:
        raise ValueError(f"No files matched pattern '{args.pattern}' under {src_path}")

    TFBExtraFeatureExtractor().process(files, args.out, verbose=args.verbose, print_every=args.print_every)


if __name__ == "__main__":
    main()

