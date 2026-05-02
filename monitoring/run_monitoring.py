"""
Component 6 — Monitoring & Drift Detection
- Generates two Evidently HTML reports (baseline vs. drift)
- Implements threshold logic (>20% feature drift → structured warning + log)
- Exposes five custom Prometheus metrics
All parameters come from configs/params.yaml.
"""

import datetime
import json
import logging
import pathlib

import numpy as np
import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    write_to_textfile,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Evidently Report Generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: dict,
    report_name: str,
) -> dict:
    """Generate an Evidently HTML report and return the drift results dict."""
    reports_dir = pathlib.Path(params["monitoring"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataSummaryPreset(),
        ]
    )
    snapshot = report.run(reference_data=reference, current_data=current)

    out_path = reports_dir / f"{report_name}.html"
    snapshot.save_html(str(out_path))
    log.info("Report saved to %s", out_path)

    return snapshot.dict()


def parse_drift_results(report_dict: dict, params: dict) -> tuple[float, list[str]]:
    """
    Parse the Evidently report dict to compute the fraction of drifted features
    and return a list of their names.
    """
    drifted_features = []
    total_features = 0

    try:
        metrics = report_dict.get("metrics", [])
        for m in metrics:
            if m.get("metric_name", "").startswith("ValueDrift"):
                total_features += 1
                col = m.get("config", {}).get("column")
                score = m.get("value")
                thresh = m.get("config", {}).get("threshold", 0.05)
                if col and score is not None and score < thresh:
                    drifted_features.append(col)
    except Exception as e:
        log.warning("Could not parse drift results: %s", e)
        return 0.0, []

    fraction = len(drifted_features) / total_features if total_features > 0 else 0.0
    return fraction, drifted_features


def apply_drift_threshold(
    fraction: float,
    drifted_features: list[str],
    params: dict,
    report_name: str,
) -> None:
    """
    If drift fraction exceeds the configured threshold, print a structured
    warning and append a log entry to the drift log file.
    """
    threshold = params["monitoring"]["drift_threshold"]
    log_path = pathlib.Path(params["monitoring"]["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if fraction > threshold:
        warning = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "report": report_name,
            "drift_fraction": round(fraction, 4),
            "threshold": threshold,
            "drifted_features": drifted_features,
            "action_required": "RETRAIN",
        }
        print("\n" + "=" * 60)
        print("⚠  DATA DRIFT ALERT")
        print(f"   Report      : {report_name}")
        print(f"   Drifted     : {len(drifted_features)} / "
              f"{int(len(drifted_features)/fraction) if fraction else 0} "
              f"features ({fraction:.1%})")
        print(f"   Threshold   : {threshold:.1%}")
        print(f"   Features    : {drifted_features}")
        print(f"   Action      : {warning['action_required']}")
        print("=" * 60 + "\n")

        with open(log_path, "a") as f:
            f.write(json.dumps(warning) + "\n")
        log.warning("Drift log entry written to %s", log_path)
    else:
        log.info(
            "Drift fraction %.2f%% is below threshold %.0f%% — no action needed.",
            fraction * 100,
            threshold * 100,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────────────────────────────────────────


def record_prometheus_metrics(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    params: dict,
    model_version: int = 1,
) -> None:
    """
    Compute and expose five custom Prometheus metrics from production data.
    Writes a text file for local scraping.
    """
    registry = CollectorRegistry()

    # 1. Confidence histogram (simulated — replace with real model scores)
    conf_hist = Histogram(
        "prediction_confidence_monitoring",
        "Prediction confidence scores on production batch",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        registry=registry,
    )
    rng = np.random.default_rng(42)
    for score in rng.uniform(0, 1, len(production)):
        conf_hist.observe(float(score))

    # 2. Feature histogram — MonthlyCharges
    mc_hist = Histogram(
        "feature_monthly_charges_monitoring",
        "MonthlyCharges distribution in production batch",
        buckets=[20, 40, 60, 80, 100, 120],
        registry=registry,
    )
    if "MonthlyCharges" in production.columns:
        for val in production["MonthlyCharges"].dropna():
            mc_hist.observe(float(val))

    # 3. Feature histogram — tenure
    tenure_hist = Histogram(
        "feature_tenure_monitoring",
        "Tenure (months) distribution in production batch",
        buckets=[0, 12, 24, 36, 48, 60, 72],
        registry=registry,
    )
    if "tenure" in production.columns:
        for val in production["tenure"].dropna():
            tenure_hist.observe(float(val))

    # 4. Current model version gauge
    version_gauge = Gauge(
        "model_version_monitoring",
        "Current model version in production",
        registry=registry,
    )
    version_gauge.set(model_version)

    # 5. Inference count by class
    target = params["data"]["target_column"]
    inference_counter = Counter(
        "inference_count_by_class_monitoring",
        "Simulated inference count by predicted class",
        ["predicted_class"],
        registry=registry,
    )
    if target in production.columns:
        for cls, cnt in production[target].value_counts().items():
            inference_counter.labels(predicted_class=str(cls)).inc(int(cnt))

    # Write metrics to file for Prometheus scraping
    prom_dir = pathlib.Path(params["monitoring"]["reports_dir"]).parent / "prometheus"
    prom_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = prom_dir / "metrics.prom"
    write_to_textfile(str(metrics_file), registry)
    log.info("Prometheus metrics written to %s", metrics_file)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    params = load_params()
    dp = params["data"]

    log.info("Loading reference and production datasets …")
    reference = pd.read_csv(dp["reference_path"])
    production = pd.read_csv(dp["production_path"])
    log.info("Reference: %d rows | Production: %d rows", len(reference), len(production))

    # ── Report 1: Baseline (reference vs. clean held-out) ──────────────────
    log.info("Generating BASELINE report (reference vs. test) …")
    # The test set uses transformed feature names; read raw for monitoring
    baseline_current = reference.sample(
        n=min(len(reference) // 5, len(reference)),
        random_state=params["data"]["random_seed"],
    )
    baseline_report_dict = generate_report(
        reference=reference,
        current=baseline_current,
        params=params,
        report_name="baseline_report",
    )
    frac_b, drifted_b = parse_drift_results(baseline_report_dict, params)
    log.info(
        "Baseline report: %.1f%% feature drift (%s)",
        frac_b * 100,
        drifted_b or "none",
    )
    apply_drift_threshold(frac_b, drifted_b, params, "baseline_report")

    # ── Report 2: Drift (reference vs. perturbed production) ───────────────
    log.info("Generating DRIFT report (reference vs. production with injected drift) …")
    drift_report_dict = generate_report(
        reference=reference,
        current=production,
        params=params,
        report_name="drift_report",
    )
    frac_d, drifted_d = parse_drift_results(drift_report_dict, params)
    log.info(
        "Drift report: %.1f%% feature drift on features: %s",
        frac_d * 100,
        drifted_d,
    )
    apply_drift_threshold(frac_d, drifted_d, params, "drift_report")

    # ── Prometheus ──────────────────────────────────────────────────────────
    log.info("Recording Prometheus metrics …")
    record_prometheus_metrics(reference, production, params, model_version=1)

    log.info("Monitoring complete.")


if __name__ == "__main__":
    main()
