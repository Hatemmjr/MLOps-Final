"""
Premium Streamlit Monitoring Dashboard for MLOps Final Project.
Unifies MLflow, Evidently, Prometheus metrics, and Business KPIs
with a Glassmorphism Neon UI and advanced Plotly analytics.
"""

import json
import pathlib
import subprocess
import yaml

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import psutil
import requests
import streamlit as st
import streamlit.components.v1 as components
from mlflow.tracking import MlflowClient
from prometheus_client.parser import text_string_to_metric_families

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS Injection (Glassmorphism + Neon)
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: #f4f5f7;
        color: #333333;
    }

    /* Hide default Streamlit headers/footers */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }

    /* Custom Vodafone Metric Cards */
    .glass-metric {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-top: 3px solid #E60000; /* Vodafone Red */
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .glass-metric:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(230, 0, 0, 0.15);
    }
    .glass-metric-magenta {
        border-top: 3px solid #333333; /* Dark Grey */
    }
    .glass-metric-magenta:hover {
        box-shadow: 0 8px 15px rgba(51, 51, 51, 0.15);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E60000;
    }
    .metric-value-magenta {
        color: #333333;
    }
    .metric-subtitle {
        font-size: 0.85rem;
        color: #999999;
        margin-top: 5px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #333333 !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    h1 {
        color: #E60000 !important;
        margin-bottom: 30px !important;
    }

    /* Tabs styling */
    [data-baseweb="tab-list"] {
        gap: 20px;
    }
    [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 4px !important;
        color: #666666 !important;
        padding: 10px 20px !important;
        border: 1px solid #e0e0e0 !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: #fcebeb !important;
        color: #E60000 !important;
        border-color: #E60000 !important;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom UI Components
# ─────────────────────────────────────────────────────────────────────────────
def render_metric(title: str, value: str, subtitle: str = "", color: str = "cyan"):
    """Render a premium glassmorphic metric card."""
    color_class = "glass-metric-magenta" if color == "magenta" else ""
    val_class = "metric-value-magenta" if color == "magenta" else ""

    html = f"""
    <div class="glass-metric {color_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-value {val_class}">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


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

# Set default plotly dark template
pio.templates.default = "plotly_white"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1>Telco Churn Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'/>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🚀 Executive Overview",
            "📊 Analytics & Insights",
            "🔬 Model Comparison",
            "📈 Model Performance",
            "📉 Data Drift",
            "💻 System Health",
            "🧪 Live API Tester",
        ],
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Fetch Prometheus
# ─────────────────────────────────────────────────────────────────────────────
def fetch_prometheus():
    host = PARAMS["serving"]["host"]
    if host == "0.0.0.0":
        host = "localhost"
    port = PARAMS["monitoring"]["prometheus_port"]

    metrics_dict = {}
    try:
        r = requests.get(f"http://{host}:{port}/", timeout=2)
        if r.status_code == 200:
            for family in text_string_to_metric_families(r.text):
                metrics_dict[family.name] = family
    except requests.exceptions.RequestException:
        pass
    return metrics_dict


# ─────────────────────────────────────────────────────────────────────────────
# 1. Executive Overview
# ─────────────────────────────────────────────────────────────────────────────
if page.startswith("🚀"):
    st.title("Executive Overview")

    metrics = fetch_prometheus()

    # ── KPI Cards ──
    col1, col2, col3, col4 = st.columns(4)

    # Production Model
    exp = client.get_experiment_by_name(PARAMS["training"]["experiment_name"])
    best_auc = 0.0
    if exp:
        runs = client.search_runs(exp.experiment_id)
        if runs:
            best_auc = max(runs, key=lambda r: r.data.metrics.get("roc_auc", 0.0)).data.metrics.get(
                "roc_auc", 0.0
            )

    with col1:
        render_metric("Prod Model AUC", f"{best_auc:.4f}", "Latest version in Registry")

    # Inference Count
    churn_cnt, nochurn_cnt = 0, 0
    if "inference_count" in metrics:
        count_metrics = metrics["inference_count"].samples
        churn_cnt = sum([s.value for s in count_metrics if s.labels.get("predicted_class") == "1"])
        nochurn_cnt = sum(
            [s.value for s in count_metrics if s.labels.get("predicted_class") == "0"]
        )

    with col2:
        render_metric(
            "Total Inferences",
            f"{int(churn_cnt + nochurn_cnt):,}",
            "Real-time API Requests",
            color="magenta",
        )

    # CPU Usage
    with col3:
        render_metric("System CPU", f"{psutil.cpu_percent()}%", "Host Server Utilization")

    # Drift
    drift_status = "0.0%"
    log_path = pathlib.Path(PARAMS["monitoring"]["log_path"])
    if log_path.exists():
        with open(log_path, "r") as f:
            logs = [json.loads(line) for line in f if line.strip()]
            if logs:
                drift_status = f"{logs[-1]['drift_fraction']*100:.1f}%"
    with col4:
        render_metric(
            "Current Drift", drift_status, "Features exceeding threshold", color="magenta"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROI Simulator ──
    st.markdown("### 💰 Projected Revenue Impact")
    prod_path = pathlib.Path(PARAMS["data"]["production_path"])
    if prod_path.exists():
        df_prod = pd.read_csv(prod_path)
        if "TotalCharges" in df_prod.columns:
            actual_churners = df_prod[df_prod[PARAMS["data"]["target_column"]] == 1]
            total_risk = actual_churners["TotalCharges"].sum()

            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**Total Revenue at Risk**: `${total_risk:,.0f}`")
                success_rate = st.slider("Campaign Success Rate (%)", 5, 80, 25) / 100.0
                cost = st.slider("Cost per Contact ($)", 10, 200, 50)

                recall, precision = 0.70, 0.60
                if runs:
                    best = max(runs, key=lambda r: r.data.metrics.get("roc_auc", 0.0))
                    recall = best.data.metrics.get("recall", 0.70)
                    precision = best.data.metrics.get("precision", 0.60)

                tp = int(len(actual_churners) * recall)
                predicted = int(tp / precision)
                revenue_saved = tp * actual_churners["TotalCharges"].mean() * success_rate
                net_roi = revenue_saved - (predicted * cost)

                render_metric(
                    "Net ROI", f"${net_roi:,.0f}", f"Assuming {success_rate*100}% retention"
                )

            with c2:
                # Plot ROI sensitivity
                rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                rois = [
                    (tp * actual_churners["TotalCharges"].mean() * r) - (predicted * cost)
                    for r in rates
                ]
                fig = px.area(
                    x=[r * 100 for r in rates],
                    y=rois,
                    title="ROI Sensitivity by Success Rate",
                    labels={"x": "Campaign Success Rate (%)", "y": "Net ROI ($)"},
                    color_discrete_sequence=["#E60000"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Analytics & Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("📊"):
    st.title("Analytics & Insights")
    st.markdown("Advanced analytics on production and predicted data.")

    prod_path = pathlib.Path(PARAMS["data"]["production_path"])
    if prod_path.exists():
        df_prod = pd.read_csv(prod_path)

        c1, c2 = st.columns(2)

        with c1:
            if "Contract" in df_prod.columns:
                fig = px.sunburst(
                    df_prod,
                    path=["Contract", PARAMS["data"]["target_column"]],
                    title="Churn Breakdown by Contract Type",
                    color=PARAMS["data"]["target_column"],
                    color_continuous_scale=["#E60000", "#333333"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "PaymentMethod" in df_prod.columns:
                fig2 = px.histogram(
                    df_prod,
                    x="PaymentMethod",
                    color=PARAMS["data"]["target_column"],
                    barmode="group",
                    title="Churn by Payment Method",
                    color_discrete_sequence=["#E60000", "#333333"],
                )
                fig2.update_layout(
                    plot_bgcolor="rgba(255,255,255,0)",
                    paper_bgcolor="rgba(255,255,255,0)",
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Feature Relationships")
        c3, c4 = st.columns(2)
        with c3:
            if "MonthlyCharges" in df_prod.columns:
                fig_box = px.box(
                    df_prod,
                    x=PARAMS["data"]["target_column"],
                    y="MonthlyCharges",
                    color=PARAMS["data"]["target_column"],
                    title="Monthly Charges Distribution",
                    color_discrete_sequence=["#E60000", "#333333"],
                )
                fig_box.update_layout(
                    plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
                )
                st.plotly_chart(fig_box, use_container_width=True)
        with c4:
            cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            if all(c in df_prod.columns for c in cols):
                corr = df_prod[cols].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    title="Numeric Correlation",
                    color_continuous_scale="Reds",
                )
                fig_corr.update_layout(
                    plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("### Production Feature Distribution (Live via Prometheus)")
        metrics = fetch_prometheus()
        mc_col, tenure_col = st.columns(2)

        with mc_col:
            if "feature_monthly_charges" in metrics:
                hist = metrics["feature_monthly_charges"].samples
                buckets = {
                    float(s.labels["le"]): s.value for s in hist if "le" in s.labels and s.labels["le"] != "+Inf"
                }
                if buckets:
                    le_keys = sorted(buckets.keys())
                    counts = [
                        buckets[k] - (buckets[le_keys[i - 1]] if i > 0 else 0)
                        for i, k in enumerate(le_keys)
                    ]
                    fig3 = px.bar(
                        x=[str(k) for k in le_keys],
                        y=counts,
                        title="MonthlyCharges (Live Requests)",
                        color_discrete_sequence=["#333333"],
                    )
                    fig3.update_layout(
                        plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
                    )
                    st.plotly_chart(fig3, use_container_width=True)

        with tenure_col:
            if "feature_tenure" in metrics:
                hist = metrics["feature_tenure"].samples
                buckets = {
                    float(s.labels["le"]): s.value for s in hist if "le" in s.labels and s.labels["le"] != "+Inf"
                }
                if buckets:
                    le_keys = sorted(buckets.keys())
                    counts = [
                        buckets[k] - (buckets[le_keys[i - 1]] if i > 0 else 0)
                        for i, k in enumerate(le_keys)
                    ]
                    fig4 = px.bar(
                        x=[str(k) for k in le_keys],
                        y=counts,
                        title="Tenure (Live Requests)",
                        color_discrete_sequence=["#E60000"],
                    )
                    fig4.update_layout(
                        plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
                    )
                    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.5 Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("🔬"):
    st.title("Model Comparison")

    exp = client.get_experiment_by_name(PARAMS["training"]["experiment_name"])
    if exp:
        runs = client.search_runs(exp.experiment_id)
        if runs:
            data = []
            for r in runs:
                name = r.data.tags.get("mlflow.runName", "unnamed").split("-")[0].title()
                data.append(
                    {
                        "Model": name,
                        "AUC": r.data.metrics.get("roc_auc", 0.0),
                        "F1": r.data.metrics.get("f1", 0.0),
                        "Recall": r.data.metrics.get("recall", 0.0),
                        "Precision": r.data.metrics.get("precision", 0.0),
                    }
                )

            df_runs = pd.DataFrame(data)
            best_models = df_runs.loc[df_runs.groupby("Model")["AUC"].idxmax()].reset_index(
                drop=True
            )

            c1, c2 = st.columns(2)
            with c1:
                df_melt = best_models.melt(id_vars="Model", var_name="Metric", value_name="Score")
                fig_bar = px.bar(
                    df_melt,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Best Models by Metric",
                    color_discrete_sequence=["#E60000", "#333333", "#666666", "#999999"],
                )
                fig_bar.update_layout(
                    plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with c2:
                categories = ["AUC", "F1", "Recall", "Precision"]
                fig_radar = go.Figure()
                colors = ["#E60000", "#333333", "#666666", "#999999", "#CCCCCC", "#000000"]
                for i, row in best_models.iterrows():
                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=[row["AUC"], row["F1"], row["Recall"], row["Precision"]],
                            theta=categories,
                            fill="toself",
                            name=row["Model"],
                            line_color=colors[i % len(colors)],
                        )
                    )
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Radar Chart Comparison",
                    paper_bgcolor="rgba(255,255,255,0)",
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("No runs found in MLflow.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("📈"):
    st.title("Model Performance")

    exp = client.get_experiment_by_name(PARAMS["training"]["experiment_name"])
    if exp:
        runs = client.search_runs(exp.experiment_id)
        if runs:
            # Table of runs
            data = [
                {
                    "Name": r.data.tags.get("mlflow.runName", "unnamed"),
                    "AUC": r.data.metrics.get("roc_auc", 0.0),
                    "F1": r.data.metrics.get("f1", 0.0),
                    "Recall": r.data.metrics.get("recall", 0.0),
                    "Precision": r.data.metrics.get("precision", 0.0),
                    "Time": pd.to_datetime(r.info.start_time, unit="ms"),
                }
                for r in runs
            ]
            df_runs = pd.DataFrame(data).sort_values("Time")

            c1, c2 = st.columns([2, 1])
            with c1:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_runs["Time"],
                        y=df_runs["AUC"],
                        mode="lines+markers",
                        name="ROC AUC",
                        line=dict(color="#E60000", width=3),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_runs["Time"],
                        y=df_runs["F1"],
                        mode="lines+markers",
                        name="F1 Score",
                        line=dict(color="#333333", width=3),
                    )
                )
                fig.update_layout(
                    title="Metric Evolution Over Optuna Trials",
                    plot_bgcolor="rgba(255,255,255,0)",
                    paper_bgcolor="rgba(255,255,255,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                best = df_runs.loc[df_runs["AUC"].idxmax()]
                render_metric("Best AUC", f"{best['AUC']:.4f}", f"Run: {best['Name']}")
                render_metric(
                    "Best F1", f"{best['F1']:.4f}", "Post-threshold tuning", color="magenta"
                )
                render_metric("Best Recall", f"{best['Recall']:.4f}", "Catching churners")

                st.markdown("### Simulated Confusion Matrix")
                p = best["Precision"]
                r = best["Recall"]
                TP = 250 * r
                FN = 250 - TP
                FP = TP / p - TP if p > 0 else 0
                TN = 750 - FP
                z = [[TN, FP], [FN, TP]]
                fig_cm = px.imshow(
                    z,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=["No Churn", "Churn"],
                    y=["No Churn", "Churn"],
                    color_continuous_scale="Reds",
                    title="Estimated Matrix (1000 samples)",
                )
                fig_cm.update_layout(
                    plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            # Try to plot feature importances if XGB/LGBM is loaded locally
            st.markdown("### Global Feature Importance")
            try:
                # Get the actual model artifact path from the best run
                best_run_obj = max(runs, key=lambda r: r.data.metrics.get("roc_auc", 0.0))
                model_uri = f"runs:/{best_run_obj.info.run_id}/model"
                loaded_model = mlflow.sklearn.load_model(model_uri)

                if hasattr(loaded_model, "feature_importances_"):
                    importances = loaded_model.feature_importances_
                    # Needs feature names. Try to get them or just plot top 20
                    # Since preprocessor removes names, we might just have indices, but let's try
                    if hasattr(loaded_model, "feature_names_in_"):
                        feats = loaded_model.feature_names_in_
                    else:
                        feats = [f"Feature {i}" for i in range(len(importances))]

                    df_imp = pd.DataFrame({"Feature": feats, "Importance": importances})
                    df_imp = df_imp.sort_values("Importance", ascending=False).head(15)

                    fig_imp = px.bar(
                        df_imp,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Top 15 Important Features",
                        color_discrete_sequence=["#E60000"],
                    )
                    fig_imp.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        plot_bgcolor="rgba(255,255,255,0)",
                        paper_bgcolor="rgba(255,255,255,0)",
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Best model doesn't expose `feature_importances_`.")
            except Exception as e:
                st.info(f"Could not load feature importances: {e}")

        else:
            st.warning("No runs found in MLflow.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Data Drift
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("📉"):
    st.title("Data Degradation & Drift")

    log_path = pathlib.Path(PARAMS["monitoring"]["log_path"])
    reports_dir = pathlib.Path(PARAMS["monitoring"]["reports_dir"])

    if log_path.exists():
        with open(log_path, "r") as f:
            logs = [json.loads(line) for line in f if line.strip()]
            if logs:
                df_logs = pd.DataFrame(logs)
                df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
                fig = px.area(
                    df_logs,
                    x="timestamp",
                    y="drift_fraction",
                    color="report",
                    title="Historical Drift Fraction",
                    color_discrete_sequence=["#E60000", "#333333"],
                )
                fig.add_hline(
                    y=PARAMS["monitoring"]["drift_threshold"],
                    line_dash="dash",
                    line_color="#ff4b4b",
                    annotation_text="Alert Threshold",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interactive Evidently Reports")
    tab1, tab2 = st.tabs(["Baseline (Holdout)", "Drift (Production)"])

    with tab1:
        if (reports_dir / "baseline_report.html").exists():
            with open(reports_dir / "baseline_report.html", "r") as f:
                components.html(f.read(), height=1200, scrolling=True)
    with tab2:
        if (reports_dir / "drift_report.html").exists():
            with open(reports_dir / "drift_report.html", "r") as f:
                components.html(f.read(), height=1200, scrolling=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. System Health
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("💻"):
    st.title("System Health & Controls")

    c1, c2, c3 = st.columns(3)
    c1.metric("CPU Utilization", f"{psutil.cpu_percent(interval=1)}%")
    mem = psutil.virtual_memory()
    c2.metric("Memory Usage", f"{mem.percent}% ({mem.used / (1024**3):.1f} GB)")
    disk = psutil.disk_usage("/")
    c3.metric("Disk Storage", f"{disk.percent}% ({disk.free / (1024**3):.1f} GB Free)")

    st.divider()

    st.markdown("### 🛠️ Pipeline Controls")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='glass-metric'>", unsafe_allow_html=True)
        st.subheader("Run Drift Monitor")
        st.write("Generates new HTML reports and parses feature distributions.")
        if st.button("Trigger Monitoring"):
            with st.spinner("Running..."):
                try:
                    res = subprocess.run(
                        ["python", "monitoring/run_monitoring.py"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    st.success("Complete!")
                except Exception:
                    st.error("Failed")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='glass-metric-magenta'>", unsafe_allow_html=True)
        st.subheader("Retrain Pipeline")
        st.write("Trigger full `dvc repro` to build a new model version.")
        if st.button("Trigger DVC Pipeline"):
            with st.spinner("Running DVC Pipeline..."):
                try:
                    res = subprocess.run(
                        ["dvc", "repro"], capture_output=True, text=True, check=True
                    )
                    st.success("Complete!")
                except Exception:
                    st.error("Failed")
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Live API Tester
# ─────────────────────────────────────────────────────────────────────────────
elif page.startswith("🧪"):
    st.title("🧪 Live API Tester")
    st.markdown(
        "Fill in the customer details below and click **Run Prediction** to get a "
        "live churn verdict from the production `/predict` endpoint."
    )

    _host = PARAMS["serving"]["host"]
    if _host == "0.0.0.0":
        _host = "localhost"
    _port = PARAMS["serving"]["port"]
    api_url = f"http://{_host}:{_port}/predict"

    with st.form("predict_form"):
        st.markdown("#### 👤 Customer Demographics")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c3:
            partner = st.selectbox("Partner", ["Yes", "No"])

        c4, c5 = st.columns(2)
        with c4:
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        with c5:
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        st.markdown("#### 📞 Services")
        c6, c7, c8 = st.columns(3)
        with c6:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
        with c7:
            internet_service = st.selectbox(
                "Internet Service", ["Fiber optic", "DSL", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
        with c8:
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )

        c9, c10 = st.columns(2)
        with c9:
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
        with c10:
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )

        st.markdown("#### 💳 Billing & Contract")
        c11, c12, c13 = st.columns(3)
        with c11:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
        with c12:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c13:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        c14, c15 = st.columns(2)
        with c14:
            monthly_charges = st.slider(
                "Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5
            )
        with c15:
            total_charges = st.slider(
                "Total Charges ($)", 0.0, 9000.0,
                float(monthly_charges * tenure), step=10.0
            )

        submitted = st.form_submit_button(
            "🚀 Run Prediction", use_container_width=True
        )

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": float(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        with st.spinner("Calling API..."):
            try:
                resp = requests.post(api_url, json=payload, timeout=5)
                resp.raise_for_status()
                result = resp.json()

                pred = result.get("prediction", -1)
                conf = result.get("confidence", 0.0)
                label = result.get("label", "Unknown")

                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")
                res_col1, res_col2 = st.columns(2)

                if pred == 1:
                    res_col1.error(
                        "**⚠️ CHURN RISK DETECTED**\n\n"
                        "This customer is predicted to churn."
                    )
                else:
                    res_col1.success(
                        "**✅ LOW CHURN RISK**\n\n"
                        "This customer is likely to stay."
                    )

                with res_col2:
                    render_metric(
                        "Model Confidence",
                        f"{conf * 100:.1f}%",
                        f"Raw label: {label}",
                        color="magenta" if pred == 1 else "cyan",
                    )

                with st.expander("📤 Request Payload (sent to API)", expanded=False):
                    st.json(payload)

                with st.expander("📥 Raw API Response", expanded=False):
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Could not reach the API at `{api_url}`.\n\n"
                    "Make sure the FastAPI server is running:\n"
                    "`uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload`"
                )
            except Exception as exc:
                st.error(f"❌ API call failed: {exc}")
