"""
Run this script AFTER training to export the MLflow experiment log.
Usage: python docs/experiment_log_generator.py
Output: docs/experiment_log.csv
"""
import mlflow
import pandas as pd
import yaml

with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)

mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name(params["training"]["experiment_name"])
if experiment is None:
    print("No experiment found. Run training first.")
    exit(1)

runs = client.search_runs(experiment_ids=[experiment.experiment_id])
rows = []
for r in runs:
    row = {"run_id": r.info.run_id, "run_name": r.info.run_name, "status": r.info.status}
    row.update(r.data.params)
    row.update(r.data.metrics)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("docs/experiment_log.csv", index=False)
print(f"Exported {len(df)} runs to docs/experiment_log.csv")
