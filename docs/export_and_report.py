"""
Run this script once to:
  1. Export all MLflow runs to docs/experiment_log.csv
  2. Print the best model metrics so you can update docs/model_card.md

Usage:
    export PYTHONPATH=.
    python docs/export_and_report.py
"""
import os
import sys
import mlflow
import pandas as pd

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "telco-churn-experiment"
OUTPUT_CSV = "docs/experiment_log.csv"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    # Find experiment
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        # Try listing all experiments
        all_exps = client.search_experiments()
        print("Available experiments:")
        for e in all_exps:
            print(f"  id={e.experiment_id}  name={e.name}")
        if not all_exps:
            print("No experiments found. Run training first:")
            print("  python -m src.training.train")
            sys.exit(1)
        exp = all_exps[0]
        print(f"\nUsing experiment: {exp.name} (id={exp.experiment_id})")
    else:
        print(f"Found experiment: {exp.name} (id={exp.experiment_id})")

    # Fetch all runs
    runs_df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.roc_auc DESC"],
    )

    if runs_df.empty:
        print("No runs found. Run training first:")
        print("  python -m src.training.train")
        sys.exit(1)

    # Save CSV
    os.makedirs("docs", exist_ok=True)
    runs_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(runs_df)} runs to {OUTPUT_CSV}")

    # Print best run metrics for model_card.md
    metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
    param_cols  = [c for c in runs_df.columns if c.startswith("params.")]

    best = runs_df.iloc[0]
    print("\n" + "="*60)
    print("BEST RUN — copy these into docs/model_card.md")
    print("="*60)
    print(f"  run_id    : {best['run_id']}")
    print(f"  run_name  : {best.get('tags.mlflow.runName', 'N/A')}")
    for col in metric_cols:
        val = best.get(col)
        if pd.notna(val):
            print(f"  {col.replace('metrics.', ''):20s}: {val:.4f}")

    print("\nALL RUNS SUMMARY:")
    summary_cols = ["run_id", "tags.mlflow.runName"] + metric_cols
    summary_cols = [c for c in summary_cols if c in runs_df.columns]
    print(runs_df[summary_cols].to_string(index=False))

    print(f"\n✅ Done. Now update docs/model_card.md with the metrics above.")
    print(f"   experiment_log.csv is at: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
