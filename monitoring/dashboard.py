"""
Streamlit Monitoring Dashboard for MLOps Final Project.
Unifies MLflow, Evidently, Prometheus metrics, and Business KPIs.
"""

import json
import pathlib
import subprocess

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import yaml
from mlflow.tracking import MlflowClient
from prometheus_client.parser import text_string_to_metric_families

st.set_page_config(
    page_title="Telco Churn Monitoring",
    page_icon="📡",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Config & Data Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_params(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


PARAMS = load_params()
MLFLOW_URI = PARAMS["mlflow"]["tracking_uri"]
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📡 MLOps Command Center")
page = st.sidebar.radio(
    "Navigation",
    [
        "1. 🚦 System Overview",
        "2. 📈 Model Performance",
        "3. 📉 Data Drift",
        "4. 💰 Business Impact",
        "5. 🛠️ Controls",
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. System Overview (FastAPI + Prometheus)
# ─────────────────────────────────────────────────────────────────────────────
if page.startswith("1"):
    st.title("🚦 System Health & API Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    # ── Fetch from FastAPI /health ──
    try:
        host = PARAMS["serving"]["host"]
        if host == "0.0.0.0":
            host = "localhost"
        port = PARAMS["serving"]["port"]
        r = requests.get(f"http://{host}:{port}/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            col1.metric("API Status", "✅ Online")
            col2.metric("Active Model", data.get("model_name", "Unknown"))
            col3.metric("Model Stage", data.get("model_version", "Unknown"))
        else:
            col1.metric("API Status", "❌ Error")
            st.error(f"Healthcheck failed: {r.status_code}")
    except requests.exceptions.RequestException:
        col1.metric("API Status", "❌ Offline")
        st.warning(f"Could not connect to FastAPI at http://{host}:{port}/health. Is it running?")

    st.divider()
    
    # ── Fetch Prometheus Metrics ──
    st.subheader("Real-time Inference Metrics")
    try:
        prom_port = PARAMS["monitoring"]["prometheus_port"]
        r = requests.get(f"http://{host}:{prom_port}/", timeout=2)
        
        if r.status_code == 200:
            # Parse Prometheus text format
            metrics_dict = {}
            for family in text_string_to_metric_families(r.text):
                metrics_dict[family.name] = family
                
            colA, colB = st.columns(2)
            
            # Inference Count Counter
            if "inference_count" in metrics_dict:
                count_metrics = metrics_dict["inference_count"].samples
                churn_cnt = sum([s.value for s in count_metrics if s.labels.get("predicted_class") == "1"])
                nochurn_cnt = sum([s.value for s in count_metrics if s.labels.get("predicted_class") == "0"])
                
                fig = px.pie(
                    values=[nochurn_cnt, churn_cnt], 
                    names=["No Churn", "Churn"],
                    title="Predictions Made",
                    hole=0.4,
                    color_discrete_sequence=["#00CC96", "#EF553B"]
                )
                colA.plotly_chart(fig, use_container_width=True)
            
            # Confidence Histogram
            if "prediction_confidence" in metrics_dict:
                hist_samples = metrics_dict["prediction_confidence"].samples
                # Extract buckets
                buckets = {}
                for s in hist_samples:
                    if "le" in s.labels and s.labels["le"] != "+Inf":
                        buckets[float(s.labels["le"])] = s.value
                
                if buckets:
                    # Convert cumulative to absolute
                    le_keys = sorted(buckets.keys())
                    counts = []
                    prev = 0
                    for k in le_keys:
                        counts.append(buckets[k] - prev)
                        prev = buckets[k]
                        
                    fig2 = px.bar(
                        x=[str(k) for k in le_keys], 
                        y=counts,
                        labels={"x": "Confidence <= x", "y": "Count"},
                        title="Prediction Confidence Distribution"
                    )
                    colB.plotly_chart(fig2, use_container_width=True)
                    
        else:
            st.error(f"Failed to fetch metrics: {r.status_code}")
    except requests.exceptions.RequestException:
        st.warning(f"Prometheus metrics endpoint (http://{host}:{prom_port}/) is unreachable.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Performance (MLflow)
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("2"):
    st.title("📈 Model Performance KPIs")
    
    # Get all experiments
    try:
        exp = client.get_experiment_by_name(PARAMS["training"]["experiment_name"])
        if exp:
            runs = client.search_runs(exp.experiment_id)
            if not runs:
                st.warning("No MLflow runs found.")
            else:
                data = []
                for r in runs:
                    metrics = r.data.metrics
                    params = r.data.params
                    tags = r.data.tags
                    
                    data.append({
                        "Run ID": r.info.run_id,
                        "Name": tags.get("mlflow.runName", "unnamed"),
                        "Status": r.info.status,
                        "AUC": metrics.get("roc_auc", 0.0),
                        "F1": metrics.get("f1", 0.0),
                        "Recall": metrics.get("recall", 0.0),
                        "Precision": metrics.get("precision", 0.0),
                        "Threshold": metrics.get("threshold", 0.0),
                        "Time": pd.to_datetime(r.info.start_time, unit="ms")
                    })
                
                df_runs = pd.DataFrame(data).sort_values("Time")
                
                # Show Best Run overall
                best_run = df_runs.loc[df_runs["AUC"].idxmax()]
                
                st.subheader(f"Best Model (AUC: {best_run['AUC']:.4f})")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ROC AUC", f"{best_run['AUC']:.4f}")
                c2.metric("F1 Score", f"{best_run['F1']:.4f}")
                c3.metric("Recall", f"{best_run['Recall']:.4f}")
                c4.metric("Opt. Threshold", f"{best_run['Threshold']:.2f}")
                
                st.divider()
                
                # Trend charts
                st.subheader("Training History Trends")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_runs["Time"], y=df_runs["AUC"], mode='lines+markers', name='ROC AUC'))
                fig.add_trace(go.Scatter(x=df_runs["Time"], y=df_runs["F1"], mode='lines+markers', name='F1 Score'))
                fig.add_trace(go.Scatter(x=df_runs["Time"], y=df_runs["Recall"], mode='lines+markers', name='Recall'))
                fig.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("All Runs")
                st.dataframe(df_runs.sort_values("AUC", ascending=False).drop(columns=["Time"]), hide_index=True)
                
        else:
            st.error(f"Experiment '{PARAMS['training']['experiment_name']}' not found in MLflow.")
            
    except Exception as e:
        st.error(f"Error connecting to MLflow: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Drift (Evidently)
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("3"):
    st.title("📉 Degradation & Data Drift")
    
    log_path = pathlib.Path(PARAMS["monitoring"]["log_path"])
    reports_dir = pathlib.Path(PARAMS["monitoring"]["reports_dir"])
    
    # Read Drift Log
    st.subheader("Drift Alert History")
    if log_path.exists():
        with open(log_path, "r") as f:
            logs = [json.loads(line) for line in f if line.strip()]
            
        if logs:
            df_logs = pd.DataFrame(logs)
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
            
            fig = px.bar(
                df_logs, 
                x="timestamp", 
                y="drift_fraction", 
                color="report",
                title="Drift Fraction History (>20% threshold)",
                labels={"drift_fraction": "Drifted Features %"}
            )
            fig.add_hline(y=PARAMS["monitoring"]["drift_threshold"], line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_logs)
        else:
            st.success("No drift alerts recorded yet.")
    else:
        st.info("Drift log file does not exist yet. Run the monitoring job.")

    st.divider()

    # Embed HTML reports
    st.subheader("Evidently Interactive Reports")
    tab1, tab2 = st.tabs(["Baseline (Test Set)", "Drift (Production Set)"])
    
    with tab1:
        baseline_html = reports_dir / "baseline_report.html"
        if baseline_html.exists():
            with open(baseline_html, "r") as f:
                html_data = f.read()
            components.html(html_data, height=1000, scrolling=True)
        else:
            st.warning("baseline_report.html not found.")
            
    with tab2:
        drift_html = reports_dir / "drift_report.html"
        if drift_html.exists():
            with open(drift_html, "r") as f:
                html_data = f.read()
            components.html(html_data, height=1000, scrolling=True)
        else:
            st.warning("drift_report.html not found.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Business Impact
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("4"):
    st.title("💰 Business ROI Simulator")
    st.markdown("Simulate the financial impact of the model on the production cohort.")
    
    prod_path = pathlib.Path(PARAMS["data"]["production_path"])
    if prod_path.exists():
        df_prod = pd.read_csv(prod_path)
        target = PARAMS["data"]["target_column"]
        
        if target in df_prod.columns and "TotalCharges" in df_prod.columns:
            actual_churners = df_prod[df_prod[target] == 1]
            total_risk = actual_churners["TotalCharges"].sum()
            
            st.info(f"**Current Production Cohort Size**: {len(df_prod):,} customers")
            st.warning(f"**Total Revenue at Risk (Actual Churners)**: ${total_risk:,.2f}")
            
            st.divider()
            
            st.subheader("Retention Campaign Simulation")
            col1, col2 = st.columns(2)
            
            success_rate = col1.slider(
                "Expected Retention Campaign Success Rate (%)", 
                min_value=5, max_value=80, value=25, step=5
            ) / 100.0
            
            cost_per_contact = col2.number_input(
                "Cost to contact/incentivize one customer ($)", 
                min_value=0.0, max_value=500.0, value=50.0, step=10.0
            )
            
            # Using MLflow best model stats to simulate predictions (simplified)
            # In a real setup, we would run df_prod through the live model API.
            # Here, we approximate based on the known Recall / Precision of the best run
            exp = client.get_experiment_by_name(PARAMS["training"]["experiment_name"])
            runs = client.search_runs(exp.experiment_id)
            if runs:
                best_run = max(runs, key=lambda r: r.data.metrics.get("roc_auc", 0.0))
                recall = best_run.data.metrics.get("recall", 0.70)
                precision = best_run.data.metrics.get("precision", 0.60)
            else:
                recall = 0.70
                precision = 0.60
                
            predicted_churn_count = int((len(actual_churners) * recall) / precision)
            true_positives = int(len(actual_churners) * recall)
            
            # Average charge of a churner
            avg_churn_val = actual_churners["TotalCharges"].mean()
            
            gross_saved = true_positives * avg_churn_val * success_rate
            campaign_cost = predicted_churn_count * cost_per_contact
            net_roi = gross_saved - campaign_cost
            
            st.markdown("### Simulated Outcomes")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Churners to Contact", f"{predicted_churn_count:,}")
            c2.metric("Campaign Cost", f"${campaign_cost:,.2f}")
            
            if net_roi > 0:
                c3.metric("Net Revenue Saved (ROI)", f"${net_roi:,.2f}", "+ Profit")
            else:
                c3.metric("Net Revenue Saved (ROI)", f"${net_roi:,.2f}", "- Loss")
                
        else:
            st.error("Production dataset must contain the target column and 'TotalCharges'.")
    else:
        st.warning("Production data not found.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Controls
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("5"):
    st.title("🛠️ Model Controls")
    
    st.markdown("Use these controls to manually trigger pipeline actions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Run Drift Monitoring")
        st.write("Generates new Evidently reports and Prometheus metrics based on the current data.")
        if st.button("Trigger Monitoring Run", type="primary"):
            with st.spinner("Running monitoring script..."):
                try:
                    result = subprocess.run(
                        ["python", "monitoring/run_monitoring.py"],
                        capture_output=True, text=True, check=True
                    )
                    st.success("Monitoring completed successfully!")
                    st.code(result.stdout)
                except subprocess.CalledProcessError as e:
                    st.error("Monitoring failed.")
                    st.code(e.stderr)

    with col2:
        st.subheader("Retrain Model Pipeline")
        st.write("Triggers the full DVC pipeline to retrain and register a new model.")
        if st.button("Trigger Full Pipeline", type="secondary"):
            with st.spinner("Running DVC pipeline (this may take a while)..."):
                try:
                    result = subprocess.run(
                        ["dvc", "repro"],
                        capture_output=True, text=True, check=True
                    )
                    st.success("Pipeline completed successfully!")
                    st.code(result.stdout)
                except subprocess.CalledProcessError as e:
                    st.error("Pipeline failed.")
                    st.code(e.stderr)
