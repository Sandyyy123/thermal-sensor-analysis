#!/usr/bin/env python3
"""Building Thermal Dynamics Analyzer - entry point."""
import argparse
import sys
from thermal_analyzer import ThermalAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Multi-sensor building thermal dynamics analysis")
    parser.add_argument("--data", required=True, help="Path to sensor CSV or directory of CSVs")
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--resample", default="5T", help="Resample interval (default: 5T = 5 minutes)")
    parser.add_argument("--anomaly-threshold", type=float, default=3.0, help="IQR multiplier for anomaly flagging")
    args = parser.parse_args()

    analyzer = ThermalAnalyzer(
        data_path=args.data,
        output_dir=args.output,
        resample_freq=args.resample,
        anomaly_threshold=args.anomaly_threshold
    )
    analyzer.run()

if __name__ == "__main__":
    main()
