import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

warnings.filterwarnings("ignore")

DEFAULT_PERIODS = [4, 7, 12, 24, 48, 52, 96, 144, 168, 336, 672, 1008, 1440]


class AlignedTFBExtractor:
    """
    TFB extractor that keeps the final benchmark pipeline shape, but computes
    Seasonality/Trend with the same logic style as Characteristics_Extractor.py.
    """

    def read_tfb_data(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        n_points = data.iloc[:, 2].value_counts().max()
        cols_name = data["cols"].unique()
        df = pd.DataFrame()
        df["date"] = data.iloc[:n_points, 0]
        for j, name in enumerate(cols_name):
            df[name] = data.iloc[j * n_points : (j + 1) * n_points, 1].values
        return df.set_index("date")

    def adjust_period(self, period_value: int) -> int:
        if abs(period_value - 4) <= 1:
            return 4
        if abs(period_value - 7) <= 1:
            return 7
        if abs(period_value - 12) <= 2:
            return 12
        if abs(period_value - 24) <= 3:
            return 24
        if abs(period_value - 48) <= 1 or ((48 - period_value) <= 4 and (48 - period_value) >= 0):
            return 48
        if abs(period_value - 52) <= 2:
            return 52
        if abs(period_value - 96) <= 10:
            return 96
        if abs(period_value - 144) <= 10:
            return 144
        if abs(period_value - 168) <= 10:
            return 168
        if abs(period_value - 336) <= 50:
            return 336
        if abs(period_value - 672) <= 20:
            return 672
        if abs(period_value - 720) <= 20:
            return 720
        if abs(period_value - 1008) <= 100:
            return 1008
        if abs(period_value - 1440) <= 200:
            return 1440
        if abs(period_value - 8766) <= 500:
            return 8766
        if abs(period_value - 10080) <= 500:
            return 10080
        if abs(period_value - 21600) <= 2000:
            return 21600
        if abs(period_value - 43200) <= 2000:
            return 43200
        return period_value

    def fft_transfer(self, timeseries: np.ndarray, fmin: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        yf = abs(np.fft.fft(timeseries))
        yfnormlize = yf / len(timeseries)
        yfhalf = yfnormlize[: len(timeseries) // 2] * 2

        fwbest = yfhalf[argrelextrema(yfhalf, np.greater)]
        xwbest = argrelextrema(yfhalf, np.greater)
        fwbest = fwbest[fwbest >= fmin].copy()

        return len(timeseries) / xwbest[0][: len(fwbest)], fwbest

    def extract_macro_stats_aligned(self, s: pd.Series) -> Dict[str, float]:
        s_clean = pd.to_numeric(s, errors="coerce").dropna()
        if len(s_clean) < 100:
            return {
                "Seasonality": 0.0,
                "Trend": 0.0,
                "Stationarity": 1.0,
                "period_value1": 0,
                "seasonal_strength1": -1.0,
                "trend_strength1": -1.0,
                "period_value2": 0,
                "seasonal_strength2": -1.0,
                "trend_strength2": -1.0,
                "period_value3": 0,
                "seasonal_strength3": -1.0,
                "trend_strength3": -1.0,
                "length": float(len(s_clean)),
            }

        try:
            adf_res = adfuller(s_clean.values, autolag="AIC")
            adf_p = float(adf_res[1])
        except Exception:
            adf_p = 1.0

        periods, amplitude = self.fft_transfer(s_clean.values, fmin=0)
        if len(amplitude) == 0:
            periods_list: List[int] = []
        else:
            order = np.argsort(amplitude)[::-1]
            periods_list = [int(round(float(periods[i]))) for i in order]

        final_periods1: List[int] = []
        for p in periods_list:
            p_adj = self.adjust_period(p)
            if p_adj not in final_periods1 and p_adj >= 4:
                final_periods1.append(p_adj)

        periods_num = min(len(final_periods1), 3)
        new_final_periods = final_periods1[:periods_num] + DEFAULT_PERIODS

        final_periods: List[int] = []
        for p in new_final_periods:
            if p not in final_periods and p >= 4:
                final_periods.append(p)

        yuzhi = max(int(len(s_clean) / 3), 12)
        season_rows: List[Tuple[float, List[float]]] = []

        for period_value in final_periods:
            if period_value < yuzhi:
                try:
                    res = STL(s_clean, period=period_value).fit()
                    temp_df = pd.DataFrame(
                        {
                            "original": s_clean,
                            "trend": res.trend,
                            "seasonal": res.seasonal,
                            "resid": res.resid,
                        }
                    )
                    temp_df["detrend"] = temp_df["original"] - temp_df["trend"]
                    temp_df["deseasonal"] = temp_df["original"] - temp_df["seasonal"]

                    den_trend = temp_df["deseasonal"].var()
                    den_season = temp_df["detrend"].var()

                    trend_strength = 0.0 if den_trend == 0 else max(0.0, 1 - temp_df["resid"].var() / den_trend)
                    seasonal_strength = 0.0 if den_season == 0 else max(0.0, 1 - temp_df["resid"].var() / den_season)

                    season_rows.append(
                        (
                            float(seasonal_strength),
                            [float(period_value), float(seasonal_strength), float(trend_strength)],
                        )
                    )
                except Exception:
                    pass

        if len(season_rows) < 3:
            for i in range(3 - len(season_rows)):
                season_rows.append((0.1 * (i + 1), [0.0, -1.0, -1.0]))

        season_rows = sorted(season_rows, key=lambda x: x[0], reverse=True)

        top = [v for _, v in season_rows[:3]]
        while len(top) < 3:
            top.append([0.0, -1.0, -1.0])

        max_seasonal_strength = top[0][1]
        max_trend_strength = top[0][2]

        return {
            "Seasonality": float(max_seasonal_strength),
            "Trend": float(max_trend_strength),
            "Stationarity": float(adf_p),
            "period_value1": int(top[0][0]),
            "seasonal_strength1": float(top[0][1]),
            "trend_strength1": float(top[0][2]),
            "period_value2": int(top[1][0]),
            "seasonal_strength2": float(top[1][1]),
            "trend_strength2": float(top[1][2]),
            "period_value3": int(top[2][0]),
            "seasonal_strength3": float(top[2][1]),
            "trend_strength3": float(top[2][2]),
            "length": float(len(s_clean)),
        }

    def process(self, files: List[str], out_dir: str, verbose: bool = False, print_every: int = 20) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        detail_dir = out_path / "dataset_details"
        detail_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: List[Dict[str, Any]] = []
        detail_rows_all: List[Dict[str, Any]] = []

        for f_path in tqdm(files, desc="Aligned Benchmarking"):
            prefix = os.path.splitext(os.path.basename(f_path))[0]
            if verbose:
                tqdm.write(f"[INFO] start dataset: {prefix}")
            try:
                df = self.read_tfb_data(f_path)
            except Exception as exc:
                print(f"[WARN] skip unreadable file: {f_path} | {exc}")
                continue

            turbines: Dict[str, List[str]] = {}
            for c in df.columns:
                tid = str(c).split(":")[0]
                turbines.setdefault(tid, []).append(c)

            stats_s: List[float] = []
            stats_t: List[float] = []
            stats_st: List[float] = []
            dataset_detail_rows: List[Dict[str, Any]] = []

            for idx, (tid, cols) in enumerate(turbines.items(), start=1):
                power_col = cols[0]
                macro = self.extract_macro_stats_aligned(df[power_col])

                stats_s.append(macro["Seasonality"])
                stats_t.append(macro["Trend"])
                stats_st.append(macro["Stationarity"])

                one_row: Dict[str, Any] = {
                    "Dataset": prefix,
                    "TurbID": tid,
                    "TargetColumn": power_col,
                    **macro,
                }
                dataset_detail_rows.append(one_row)
                detail_rows_all.append(one_row)

                if verbose and (idx == 1 or idx % max(1, int(print_every)) == 0 or idx == len(turbines)):
                    tqdm.write(
                        f"[INFO] {prefix} | turb {idx}/{len(turbines)} | "
                        f"TurbID={tid} | S={macro['Seasonality']:.4f} T={macro['Trend']:.4f}"
                    )

            summary_rows.append(
                {
                    "Dataset": prefix,
                    "Seasonality": float(np.nanmean(stats_s)) if stats_s else np.nan,
                    "Trend": float(np.nanmean(stats_t)) if stats_t else np.nan,
                    "Stationarity": float(np.nanmean(stats_st)) if stats_st else np.nan,
                    "NumTurbines": int(len(dataset_detail_rows)),
                }
            )

            if dataset_detail_rows:
                pd.DataFrame(dataset_detail_rows).to_csv(detail_dir / f"{prefix}_detail.csv", index=False)
                if verbose:
                    tqdm.write(
                        f"[INFO] done dataset: {prefix} | turbines={len(dataset_detail_rows)} | "
                        f"mean_S={np.nanmean(stats_s):.4f} mean_T={np.nanmean(stats_t):.4f}"
                    )

        pd.DataFrame(summary_rows).to_csv(out_path / "Final_Benchmark_Report.csv", index=False)
        pd.DataFrame(detail_rows_all).to_csv(out_path / "Dataset_Detail_Report.csv", index=False)
        print(f"[INFO] summary saved: {out_path / 'Final_Benchmark_Report.csv'}")
        print(f"[INFO] details saved: {out_path / 'Dataset_Detail_Report.csv'}")
        print(f"[INFO] per-dataset details dir: {detail_dir}")


def _run_self_test() -> None:
    root = Path("./_aligned_extractor_self_test")
    src = root / "src"
    out = root / "out"
    src.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    n = 300
    t = pd.date_range("2020-01-01", periods=n, freq="10min")
    rows = []

    # Build a tiny tfb long-table file with 2 turbines x 2 vars.
    for name in [
        "type 1:Power (kW)",
        "type 2:Power (kW)",
        "type 1:Wind speed (m/s)",
        "type 2:Wind speed (m/s)",
    ]:
        if "Power" in name:
            data = np.sin(np.arange(n) / 24.0) + (1.0 if "type 2" in name else 0.0)
        else:
            data = np.cos(np.arange(n) / 24.0) + 5.0
        rows.append(pd.DataFrame({"date": t, "data": data, "cols": name}))

    tfb_path = src / "tfb-selftest.csv"
    pd.concat(rows, ignore_index=True).to_csv(tfb_path, index=False)

    extractor = AlignedTFBExtractor()
    extractor.process([str(tfb_path)], str(out))

    assert (out / "Final_Benchmark_Report.csv").exists()
    assert (out / "Dataset_Detail_Report.csv").exists()
    assert (out / "dataset_details" / "tfb-selftest_detail.csv").exists()
    print("[INFO] self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TFB extractor with season/trend aligned to Characteristics_Extractor.py logic."
    )
    parser.add_argument("--src", default="/home/dag-test-zhu/dataset/forecasting_wp1", help="Source directory")
    parser.add_argument("--out", default="/home/dag-test-zhu/dataset/wind_final_stats_aligned_compare", help="Output directory")
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

    AlignedTFBExtractor().process(files, args.out, verbose=args.verbose, print_every=args.print_every)


if __name__ == "__main__":
    main()

