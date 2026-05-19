"""Core thermal analysis logic."""
import os
import pandas as pd
import numpy as np
from scipy import signal, stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")


class ThermalAnalyzer:
    def __init__(self, data_path: str, output_dir: str, resample_freq: str = "5T", anomaly_threshold: float = 3.0):
        self.data_path = data_path
        self.output_dir = output_dir
        self.resample_freq = resample_freq
        self.anomaly_threshold = anomaly_threshold
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        print("=== Building Thermal Dynamics Analysis ===")
        df = self.load_data()
        df = self.preprocess(df)
        anomalies = self.detect_anomalies(df)
        thermal_response = self.analyze_thermal_response(df)
        self.compute_correlations(df)
        self.save_results(df, anomalies, thermal_response)
        print(f"Done. Results saved to {self.output_dir}")

    def load_data(self) -> pd.DataFrame:
        """Load sensor CSV(s) with auto-detection of timestamp column."""
        print("Loading sensor data...")
        if os.path.isdir(self.data_path):
            dfs = []
            for f in os.listdir(self.data_path):
                if f.endswith(".csv"):
                    dfs.append(pd.read_csv(os.path.join(self.data_path, f)))
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            df = pd.read_csv(self.data_path)

        # Auto-detect timestamp column
        ts_cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "ts", "stamp"])]
        if ts_cols:
            df[ts_cols[0]] = pd.to_datetime(df[ts_cols[0]], infer_datetime_format=True, utc=True)
            df = df.set_index(ts_cols[0]).sort_index()
        print(f"  Loaded {len(df)} rows, {df.shape[1]} sensor columns")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample, interpolate, detect drift."""
        print("Preprocessing...")
        # Resample to uniform grid
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols].resample(self.resample_freq).mean()

        # Report completeness per sensor
        completeness = (df.notna().sum() / len(df) * 100).round(1)
        print("  Data completeness per sensor:")
        for col, pct in completeness.items():
            flag = "OK" if pct >= 90 else "WARN" if pct >= 75 else "CRITICAL"
            print(f"    [{flag}] {col}: {pct}%")

        # Interpolate gaps up to 2h (24 steps at 5min)
        df = df.interpolate(method="time", limit=24, limit_direction="both")

        # Detect linear drift using Theil-Sen slope
        for col in df.columns:
            series = df[col].dropna()
            if len(series) > 50:
                x = np.arange(len(series))
                slope, _, _, _ = stats.theilslopes(series.values, x)
                if abs(slope) > 0.001:  # threshold per interval
                    print(f"  Drift detected in {col}: {slope:.4f} units/interval")

        return df

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR-based + isolation forest anomaly detection."""
        print("Detecting anomalies...")
        anomaly_records = []

        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 20:
                continue

            # IQR method
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - self.anomaly_threshold * iqr, q3 + self.anomaly_threshold * iqr
            iqr_flags = (series < lo) | (series > hi)

            # Isolation forest
            iso = IsolationForest(contamination=0.02, random_state=42)
            iso_labels = iso.fit_predict(series.values.reshape(-1, 1))
            iso_flags = pd.Series(iso_labels == -1, index=series.index)

            # Combine: flag if EITHER method triggers
            combined = iqr_flags | iso_flags
            n_flagged = combined.sum()
            if n_flagged > 0:
                print(f"  {col}: {n_flagged} anomalies flagged")
                for ts in combined[combined].index:
                    anomaly_records.append({
                        "timestamp": ts, "sensor": col,
                        "value": df.loc[ts, col],
                        "iqr_flag": bool(iqr_flags.get(ts, False)),
                        "iso_flag": bool(iso_flags.get(ts, False))
                    })

        return pd.DataFrame(anomaly_records)

    def analyze_thermal_response(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute thermal time constants and response curves per heating event."""
        print("Analyzing thermal response...")
        temp_cols = [c for c in df.columns if "temp" in c.lower() or "temperature" in c.lower()]
        heat_cols = [c for c in df.columns if any(k in c.lower() for k in ["heat", "flow", "return", "hvac"])]

        if not temp_cols or not heat_cols:
            print("  No temp/heating columns detected; skipping response analysis")
            return pd.DataFrame()

        results = []
        for tc in temp_cols:
            series = df[tc].dropna()
            # Find heating ON events via derivative threshold
            deriv = series.diff().fillna(0)
            heating_starts = series.index[deriv > deriv.std() * 2]

            for start in heating_starts[:20]:  # cap at 20 events
                try:
                    window = series[start:].iloc[:120]  # 10h window at 5min
                    if len(window) < 10:
                        continue
                    t_start = window.iloc[0]
                    t_max = window.max()
                    delta = t_max - t_start
                    if delta < 0.5:
                        continue
                    # T50 and T90
                    t50_idx = (window >= t_start + 0.5 * delta).idxmax()
                    t90_idx = (window >= t_start + 0.9 * delta).idxmax()
                    t50_min = (t50_idx - start).total_seconds() / 60
                    t90_min = (t90_idx - start).total_seconds() / 60
                    results.append({"sensor": tc, "event_start": start, "delta_T": round(delta, 2),
                                    "T50_min": round(t50_min, 1), "T90_min": round(t90_min, 1)})
                except Exception:
                    continue

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            print(f"  Analyzed {len(result_df)} heating events across {result_df.sensor.nunique()} sensors")
        return result_df

    def compute_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-correlation matrix between sensors."""
        corr = df.corr(method="pearson")
        print("  Correlation matrix computed")
        corr.to_csv(os.path.join(self.output_dir, "correlations.csv"))
        return corr

    def save_results(self, df, anomalies, thermal_response):
        df.to_csv(os.path.join(self.output_dir, "cleaned_data.csv"))
        if not anomalies.empty:
            anomalies.to_csv(os.path.join(self.output_dir, "anomalies.csv"), index=False)
        if not thermal_response.empty:
            thermal_response.to_csv(os.path.join(self.output_dir, "thermal_response.csv"), index=False)
        print(f"  All outputs saved to {self.output_dir}")
