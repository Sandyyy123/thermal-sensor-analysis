# thermal-sensor-analysis

Multi-sensor building thermal dynamics analysis pipeline.

## Overview

Analyzes real thermal behavior of residential buildings from multi-sensor time-series data:
- Temperature sensors (per room)
- Humidity sensors
- Heating system signals (flow/return temperatures)
- External climate data

## Pipeline

```
Raw CSV/JSON sensors
    → Preprocessing (imputation, resampling, drift correction)
    → Anomaly detection (IQR + LOWESS + isolation forest)
    → Thermal response modeling (T50/T90, time constants)
    → Correlation & regression analysis
    → Plotly interactive dashboard + PDF report
```

## Setup

```bash
pip install -r requirements.txt
python main.py --data path/to/sensors.csv --output results/
```

## Output

- `results/cleaned_data.csv` - merged, imputed, quality-flagged sensor data
- `results/anomalies.csv` - flagged events with root-cause labels
- `results/thermal_response.csv` - T50, T90, tau per room per heating event
- `results/dashboard.html` - interactive Plotly dashboard
- `results/report.pdf` - structured analysis report
