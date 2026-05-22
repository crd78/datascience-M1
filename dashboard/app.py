from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import (  # noqa: E402
    BEST_MODEL_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_TARGET,
    FEATURE_IMPORTANCE_PATH,
    LAG_STEPS,
    METRICS_CSV_PATH,
    REPORTS_DIR,
    TARGET_HORIZON_HOURS,
)
from src.data import add_temporal_features, load_dataset  # noqa: E402
from src.forecast import forecast_future_failures  # noqa: E402
from src.postprocessing import (  # noqa: E402
    add_persistent_risk_alerts,
    estimate_sampling_interval_minutes,
)
from src.predict import load_artifact, predict_one  # noqa: E402


st.set_page_config(page_title="Maintenance predictive industrielle", layout="wide")

COLOR_OK = "#15803d"
COLOR_WARN = "#b45309"
COLOR_RISK = "#b91c1c"
COLOR_TEXT = "#111827"
COLOR_MUTED = "#6b7280"
COLOR_PANEL = "#ffffff"
COLOR_BG = "#f5f7fb"
COLOR_BORDER = "#d8dee9"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {COLOR_BG};
            color: {COLOR_TEXT};
        }}
        h1, h2, h3 {{
            color: {COLOR_TEXT};
            letter-spacing: 0;
        }}
        [data-testid="stMetric"] {{
            background: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 14px 16px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {COLOR_MUTED};
        }}
        .section-note {{
            color: {COLOR_MUTED};
            font-size: 0.95rem;
            margin-top: -0.4rem;
            margin-bottom: 1rem;
        }}
        .status-pill {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            border: 1px solid {COLOR_BORDER};
            background: #fff;
        }}
        .pill-risk {{
            color: {COLOR_RISK};
            background: #fef2f2;
            border-color: #fecaca;
        }}
        .pill-watch {{
            color: {COLOR_WARN};
            background: #fffbeb;
            border-color: #fde68a;
        }}
        .pill-ok {{
            color: {COLOR_OK};
            background: #f0fdf4;
            border-color: #bbf7d0;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_default_dataset() -> Path:
    return DEFAULT_DATA_PATH


@st.cache_data(show_spinner=False)
def cached_load_dataset(path: str) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_data(show_spinner=False)
def cached_engineer_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return add_temporal_features(df, target)[0]


@st.cache_resource(show_spinner=False)
def cached_load_artifact(path: str) -> dict[str, Any]:
    return load_artifact(path)


@st.cache_data(show_spinner=False)
def cached_metrics(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_importance(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def risk_label(risk: float, threshold: float) -> str:
    if risk >= threshold:
        return "Critique"
    if risk >= threshold * 0.6:
        return "A surveiller"
    return "Stable"


def risk_class(risk: float, threshold: float) -> str:
    if risk >= threshold:
        return "pill-risk"
    if risk >= threshold * 0.6:
        return "pill-watch"
    return "pill-ok"


def add_model_predictions(engineered_df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    df = engineered_df.copy()
    feature_columns = artifact["feature_columns"]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Features absentes pour le dashboard: {missing}")

    pipeline = artifact["pipeline"]
    df["predicted_failure"] = pipeline.predict(df[feature_columns])
    if hasattr(pipeline, "predict_proba"):
        df["risk_probability"] = pipeline.predict_proba(df[feature_columns])[:, 1]
    else:
        df["risk_probability"] = df["predicted_failure"].astype(float)
    return df


def build_simulated_history(
    raw_df: pd.DataFrame,
    machine_id: Any,
    payload: dict[str, Any],
    adjustable_features: list[str],
) -> pd.DataFrame:
    if "machine_id" not in raw_df.columns or "timestamp" not in raw_df.columns:
        return pd.DataFrame()

    history = raw_df[raw_df["machine_id"] == machine_id].copy()
    if history.empty:
        return history

    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
    history = history.dropna(subset=["timestamp"]).sort_values("timestamp")
    if history.empty:
        return history

    last_index = history.index[-1]
    for col in adjustable_features + ["operating_mode"]:
        if col in history.columns and col in payload:
            history.loc[last_index, col] = payload[col]

    return history


def latest_machine_state(scored_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in scored_df.columns:
        ordered = scored_df.sort_values(["machine_id", "timestamp"])
    else:
        ordered = scored_df.copy()

    if "machine_id" in ordered.columns:
        latest = ordered.groupby("machine_id", as_index=False).tail(1)
    else:
        latest = ordered.tail(1)

    sort_cols = [col for col in ["persistent_alert", "risk_smoothed", "risk_probability"] if col in latest.columns]
    return latest.sort_values(sort_cols, ascending=False)


def summarize_future_risk(
    scored_df: pd.DataFrame,
    future_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    if future_df.empty:
        return pd.DataFrame()

    grouped = future_df.groupby("machine_id", as_index=False)
    agg_map = {
        "max_future_risk": ("risk_smoothed", "max"),
        "peak_raw_risk": ("risk_probability", "max"),
        "future_points": ("timestamp", "count"),
        "alert_points": ("persistent_alert", "sum"),
        "last_future_timestamp": ("timestamp", "max"),
    }
    if "mode_drift_factor" in future_df.columns:
        agg_map["mode_drift_factor"] = ("mode_drift_factor", "max")
    summary = grouped.agg(**agg_map)

    alert_rows = future_df[future_df["persistent_alert"]].sort_values(["machine_id", "timestamp"])
    first_alert = alert_rows.groupby("machine_id")["timestamp"].min().rename("first_alert_at")
    summary = summary.merge(first_alert, on="machine_id", how="left")

    last_observed = scored_df.groupby("machine_id")["timestamp"].max().rename("last_observed_at")
    summary = summary.merge(last_observed, on="machine_id", how="left")
    summary["alert_in_horizon"] = summary["alert_points"] > 0
    summary["hours_until_alert"] = (
        summary["first_alert_at"] - summary["last_observed_at"]
    ).dt.total_seconds() / 3600

    latest_meta_cols = [
        col
        for col in ["machine_id", "machine_type", "operating_mode", "risk_smoothed", "persistent_alert"]
        if col in scored_df.columns
    ]
    latest_meta = latest_machine_state(scored_df)[latest_meta_cols].rename(
        columns={"risk_smoothed": "current_risk", "persistent_alert": "current_alert"}
    )
    summary = summary.merge(latest_meta, on="machine_id", how="left")

    summary["action"] = np.select(
        [
            summary["alert_in_horizon"],
            summary["max_future_risk"] >= threshold * 0.6,
        ],
        [
            "Planifier maintenance",
            "Surveiller",
        ],
        default="Stable",
    )
    return summary.sort_values(
        ["alert_in_horizon", "max_future_risk", "peak_raw_risk"],
        ascending=[False, False, False],
    )


def metric_delta_text(current_points: float, reference_points: float) -> str:
    diff = current_points - reference_points
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f} pts vs historique"


def plot_confusion_matrix(metrics_row: pd.Series) -> go.Figure:
    values = np.array(
        [
            [metrics_row.get("tn", 0), metrics_row.get("fp", 0)],
            [metrics_row.get("fn", 0), metrics_row.get("tp", 0)],
        ]
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=["Pred: pas de panne", "Pred: panne"],
            y=["Reel: pas de panne", "Reel: panne"],
            colorscale=[[0, "#eef2ff"], [1, "#1d4ed8"]],
            text=values,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "#111827"},
            showscale=False,
        )
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(side="top"))
    return fig


inject_css()

with st.sidebar:
    st.header("Parametres")
    dataset_path = str(find_default_dataset())
    st.caption(f"Dataset utilise: {dataset_path}")
    model_path = st.text_input("Modele", value=str(BEST_MODEL_PATH))
    risk_threshold = st.slider("Seuil d'alerte", 0.10, 0.90, 0.50, 0.05)
    smoothing_window = st.slider("Lissage du risque", 1, 15, 5, 1)
    persistence_points = st.slider("Persistance minimale", 1, 8, 3, 1)
    st.caption(
        "Une alerte critique est declenchee seulement si le risque lisse reste au-dessus du seuil pendant plusieurs mesures."
    )

try:
    df = cached_load_dataset(dataset_path)
except Exception as exc:
    st.error(f"Impossible de charger le dataset: {exc}")
    st.stop()

target = DEFAULT_TARGET if DEFAULT_TARGET in df.columns else None
if not target:
    st.error(f"La cible {DEFAULT_TARGET} est absente du dataset.")
    st.stop()

try:
    artifact = cached_load_artifact(model_path)
except Exception as exc:
    st.error(f"Impossible de charger le modele: {exc}")
    st.stop()

engineered_df = cached_engineer_features(df, target)
try:
    scored_df = add_model_predictions(engineered_df, artifact)
except Exception as exc:
    st.error(f"Prediction impossible pour le dashboard: {exc}")
    st.stop()

scored_df = add_persistent_risk_alerts(
    scored_df,
    threshold=risk_threshold,
    smoothing_window=smoothing_window,
    persistence_points=persistence_points,
)
latest_df = latest_machine_state(scored_df)
failure_rate = float(df[target].mean())
latest_risk_mean = float(latest_df["risk_smoothed"].mean()) if len(latest_df) else 0.0
critical_count = int(latest_df["persistent_alert"].sum())
watch_count = int(
    (
        (latest_df["risk_smoothed"] >= risk_threshold * 0.6)
        & ~latest_df["persistent_alert"]
    ).sum()
)
ignored_spikes_count = int(latest_df["ignored_spike"].sum())
machine_count = int(df["machine_id"].nunique()) if "machine_id" in df.columns else len(latest_df)
sampling_minutes = estimate_sampling_interval_minutes(scored_df)
persistence_duration = (
    f"environ {sampling_minutes * persistence_points:.0f} min"
    if sampling_minutes is not None
    else f"{persistence_points} mesures"
)

st.title("Maintenance predictive industrielle")
st.markdown(
    "<div class='section-note'>Pilotage des risques machine, priorisation des interventions et suivi de performance du modele.</div>",
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Machines suivies", machine_count)
metric_2.metric("Observations", f"{len(df):,}".replace(",", " "))
metric_3.metric("Pannes historiques", pct(failure_rate))
metric_4.metric(
    "Risque moyen lisse",
    pct(latest_risk_mean),
    metric_delta_text(latest_risk_mean * 100, failure_rate * 100),
)
metric_5.metric("Alertes persistantes", critical_count, f"{watch_count} a surveiller")
st.caption(
    f"Regle d'alerte : moyenne glissante sur {smoothing_window} mesures, puis confirmation sur {persistence_points} mesures consecutives ({persistence_duration}). Pics actuels ignores : {ignored_spikes_count}."
)

overview_tab, fleet_tab, forecast_tab, model_tab, data_tab, simulation_tab = st.tabs(
    [
        "Vue maintenance",
        "Parc machines",
        "Prevision future",
        "Performance modele",
        "Qualite donnees",
        "Simulation",
    ]
)

with overview_tab:
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("File de priorite maintenance")
        st.markdown(
            f"<div class='section-note'>Machines classees par risque lisse et alerte persistante. Une alerte doit rester haute pendant {persistence_duration}.</div>",
            unsafe_allow_html=True,
        )
        alert_columns = [
            col
            for col in [
                "machine_id",
                "machine_type",
                "timestamp",
                "operating_mode",
                "risk_smoothed",
                "risk_probability",
                "persistent_alert",
                "ignored_spike",
                "vibration_rms",
                "temperature_motor",
                "hours_since_maintenance",
            ]
            if col in latest_df.columns
        ]
        alert_table = latest_df[alert_columns].copy().head(12)
        if "risk_smoothed" in alert_table.columns:
            alert_table["risk_smoothed"] = (alert_table["risk_smoothed"] * 100).round(1)
        if "risk_probability" in alert_table.columns:
            alert_table["risk_probability"] = (alert_table["risk_probability"] * 100).round(1)
        st.dataframe(
            alert_table,
            width="stretch",
            hide_index=True,
            column_config={
                "risk_smoothed": st.column_config.ProgressColumn(
                    "Risque lisse",
                    min_value=0,
                    max_value=100,
                    format="%.1f %%",
                ),
                "risk_probability": st.column_config.ProgressColumn(
                    "Risque brut",
                    min_value=0,
                    max_value=100,
                    format="%.1f %%",
                ),
            },
        )
        st.caption(
            "Utilite : cette table priorise les interventions avec le risque lisse. Un pic brut isole peut apparaitre, mais il ne devient critique que s'il persiste."
        )

    with right:
        st.subheader("Repartition du risque actuel")
        risk_status = np.select(
            [
                latest_df["persistent_alert"],
                latest_df["risk_smoothed"] >= risk_threshold * 0.6,
            ],
            ["Critique", "A surveiller"],
            default="Stable",
        )
        risk_counts = pd.Series(risk_status).value_counts().reindex(["Stable", "A surveiller", "Critique"]).fillna(0)
        risk_plot_df = risk_counts.reset_index()
        risk_plot_df.columns = ["niveau", "machines"]
        fig = px.bar(
            risk_plot_df,
            x="niveau",
            y="machines",
            color="niveau",
            color_discrete_map={"Stable": COLOR_OK, "A surveiller": COLOR_WARN, "Critique": COLOR_RISK},
        )
        fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Utilite : ce graphique resume le parc apres lissage. Les pics isoles ne basculent plus automatiquement en critique."
        )

    st.subheader("Signaux associes aux pannes")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(
            df,
            x=target,
            y="temperature_motor",
            color=target,
            labels={target: "Panne dans 24h", "temperature_motor": "Temperature moteur"},
            color_discrete_sequence=["#2563eb", COLOR_RISK],
        )
        fig.update_layout(showlegend=False, height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Lecture : si la distribution des pannes est decalee vers des temperatures plus hautes, la temperature moteur est un signal utile pour anticiper les defaillances."
        )
    with c2:
        fig = px.box(
            df,
            x=target,
            y="vibration_rms",
            color=target,
            labels={target: "Panne dans 24h", "vibration_rms": "Vibration RMS"},
            color_discrete_sequence=["#2563eb", COLOR_RISK],
        )
        fig.update_layout(showlegend=False, height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Lecture : ce graphique compare les niveaux de vibration entre les periodes avec et sans panne. Une vibration plus elevee peut indiquer une degradation mecanique."
        )

with fleet_tab:
    st.subheader("Analyse par machine")
    machine_options = sorted(scored_df["machine_id"].dropna().unique().tolist()) if "machine_id" in scored_df.columns else []
    selected_machine = st.selectbox("Machine", machine_options, index=0 if machine_options else None)

    if selected_machine is not None:
        machine_df = scored_df[scored_df["machine_id"] == selected_machine].sort_values("timestamp")
        last_row = machine_df.tail(1).iloc[0]
        risk = float(last_row["risk_smoothed"])
        label = "Critique" if bool(last_row["persistent_alert"]) else risk_label(risk, risk_threshold)
        css_class = "pill-risk" if bool(last_row["persistent_alert"]) else risk_class(risk, risk_threshold)
        st.markdown(
            f"<span class='status-pill {css_class}'>{label} - risque lisse {risk * 100:.1f}%</span>",
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=machine_df["timestamp"],
                y=machine_df["risk_probability"],
                name="Risque brut",
                line=dict(color="#fca5a5", width=1),
                opacity=0.55,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=machine_df["timestamp"],
                y=machine_df["risk_smoothed"],
                name="Risque lisse",
                line=dict(color=COLOR_RISK, width=3),
            )
        )
        fig.add_hline(y=risk_threshold, line_dash="dash", line_color=COLOR_WARN, annotation_text="Seuil")
        fig.update_layout(yaxis_tickformat=".0%", height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Lecture : la courbe claire montre les pics bruts du modele. La courbe rouge foncee est lissee. L'alerte n'est critique que si cette courbe reste au-dessus du seuil plusieurs mesures."
        )

        sensors = [
            col
            for col in ["temperature_motor", "vibration_rms", "current_phase_avg", "pressure_level", "hours_since_maintenance"]
            if col in machine_df.columns
        ]
        selected_sensor = st.selectbox("Signal capteur", sensors)
        fig = px.line(machine_df, x="timestamp", y=selected_sensor, color=target)
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Utilite : relier un capteur precis a l'historique des pannes. Cela aide a expliquer pourquoi le risque augmente pour une machine donnee."
        )

with forecast_tab:
    st.subheader("Prevision apres le dernier historique")
    st.markdown(
        "<div class='section-note'>Projection court terme par machine : on stabilise les derniers signaux capteurs, on recalcule les lags, puis on applique le modele et la regle de persistance.</div>",
        unsafe_allow_html=True,
    )
    lag_minutes = [int(round((sampling_minutes or 0) * lag)) for lag in LAG_STEPS]
    st.caption(
        f"Limite modele : la cible est {DEFAULT_TARGET}, donc l'horizon maximum coherent est {TARGET_HORIZON_HOURS}h. "
        f"Avec la cadence mediane du dataset, les lags {LAG_STEPS} couvrent environ {lag_minutes} minutes."
    )
    st.caption("Regle de projection : la duree reste identique pour tous les modes. Idle stabilise les capteurs, normal applique la derive moyenne, peak accelere les deltas capteurs estimes depuis predictive_maintenance_v3.csv.")

    default_step = int(round(sampling_minutes or 15))
    step_options = sorted({5, 10, 15, 30, 60, default_step})
    default_step_index = step_options.index(default_step) if default_step in step_options else 0

    f1, f2, f3 = st.columns(3)
    with f1:
        horizon_hours = st.slider(
            "Horizon futur",
            1,
            TARGET_HORIZON_HOURS,
            TARGET_HORIZON_HOURS,
            1,
            help="Le modele a ete entraine pour predire failure_within_24h. Au-dela de 24h, il faut creer une nouvelle cible.",
        )
    with f2:
        forecast_step_minutes = st.selectbox(
            "Pas de projection",
            step_options,
            index=default_step_index,
            format_func=lambda value: f"{value} min",
            help="Frequence des points futurs. Par defaut, on reprend le rythme observe dans le dataset.",
        )
    with f3:
        trend_window = st.slider(
            "Mesures de reference",
            4,
            48,
            12,
            1,
            help="Nombre de dernieres mesures utilisees comme etat recent de la machine.",
        )

    f4, f5 = st.columns(2)
    with f4:
        scenario_label = st.selectbox(
            "Scenario capteurs",
            ["Etat recent + mode", "Tendance amortie"],
            index=0,
            help="Etat recent + mode applique la derive capteur moyenne du mode. Tendance amortie ajoute aussi la tendance recente de la machine.",
        )
    with f5:
        future_mode_label = st.selectbox(
            "Mode futur",
            ["Mode courant", "idle", "normal", "peak"],
            index=0,
            help="Mode courant garde le dernier mode observe de chaque machine. Les autres choix forcent un scenario metier.",
        )
    forecast_scenario = "stable_recent" if scenario_label == "Etat recent + mode" else "damped_trend"
    trend_strength = 0.0 if forecast_scenario == "stable_recent" else 0.15
    future_mode = None if future_mode_label == "Mode courant" else future_mode_label

    try:
        future_df = forecast_future_failures(
            df,
            artifact,
            horizon_hours=horizon_hours,
            step_minutes=forecast_step_minutes,
            trend_window=trend_window,
            scenario=forecast_scenario,
            trend_strength=trend_strength,
            future_mode=future_mode,
        )
        combined_forecast = pd.concat(
            [scored_df.assign(is_future=False), future_df],
            ignore_index=True,
            sort=False,
        )
        combined_forecast = add_persistent_risk_alerts(
            combined_forecast,
            threshold=risk_threshold,
            smoothing_window=smoothing_window,
            persistence_points=persistence_points,
        )
        future_scored = combined_forecast[combined_forecast["is_future"].eq(True)].copy()
        future_summary = summarize_future_risk(scored_df, future_scored, risk_threshold)
    except Exception as exc:
        st.error(f"Prevision future impossible: {exc}")
        future_scored = pd.DataFrame()
        future_summary = pd.DataFrame()

    if future_summary.empty:
        st.warning("Aucune prevision future disponible.")
    else:
        forecast_alerts = int(future_summary["alert_in_horizon"].sum())
        forecast_watch = int(
            (
                (future_summary["max_future_risk"] >= risk_threshold * 0.6)
                & ~future_summary["alert_in_horizon"]
            ).sum()
        )
        max_projected_risk = float(future_summary["max_future_risk"].max())
        next_alert = future_summary["first_alert_at"].dropna().min()
        next_alert_label = str(next_alert) if pd.notna(next_alert) else "Aucune"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Machines en alerte future", forecast_alerts)
        k2.metric("Machines a surveiller", forecast_watch)
        k3.metric("Risque futur max", pct(max_projected_risk))
        k4.metric("Prochaine alerte", next_alert_label)

        table_cols = [
            col
            for col in [
                "machine_id",
                "machine_type",
                "operating_mode",
                "mode_drift_factor",
                "action",
                "current_risk",
                "max_future_risk",
                "peak_raw_risk",
                "first_alert_at",
                "hours_until_alert",
                "future_points",
            ]
            if col in future_summary.columns
        ]
        display_summary = future_summary[table_cols].copy().head(15)
        for col in ["current_risk", "max_future_risk", "peak_raw_risk"]:
            if col in display_summary.columns:
                display_summary[col] = (display_summary[col] * 100).round(1)
        if "hours_until_alert" in display_summary.columns:
            display_summary["hours_until_alert"] = display_summary["hours_until_alert"].round(2)
        if "mode_drift_factor" in display_summary.columns:
            display_summary["mode_drift_factor"] = display_summary["mode_drift_factor"].round(2)

        st.dataframe(
            display_summary,
            width="stretch",
            hide_index=True,
            column_config={
                "current_risk": st.column_config.ProgressColumn(
                    "Risque actuel",
                    min_value=0,
                    max_value=100,
                    format="%.1f %%",
                ),
                "max_future_risk": st.column_config.ProgressColumn(
                    "Risque futur lisse max",
                    min_value=0,
                    max_value=100,
                    format="%.1f %%",
                ),
                "peak_raw_risk": st.column_config.ProgressColumn(
                    "Pic brut futur",
                    min_value=0,
                    max_value=100,
                    format="%.1f %%",
                ),
            },
        )
        st.caption(
            "Utilite : cette table donne la file de maintenance future. Le scenario stable ne force pas les capteurs a monter : une alerte future apparait seulement si l'etat recent, le temps et les lags restent coherents avec un risque durable."
        )

        forecast_machine_options = future_summary["machine_id"].dropna().tolist()
        selected_forecast_machine = st.selectbox(
            "Machine a projeter",
            forecast_machine_options,
            index=0 if forecast_machine_options else None,
            key="forecast_machine",
        )

        if selected_forecast_machine is not None:
            history_machine = scored_df[scored_df["machine_id"] == selected_forecast_machine].sort_values("timestamp").tail(160)
            future_machine = future_scored[future_scored["machine_id"] == selected_forecast_machine].sort_values("timestamp")
            last_history_ts = history_machine["timestamp"].max() if len(history_machine) else None

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=history_machine["timestamp"],
                    y=history_machine["risk_smoothed"],
                    name="Historique lisse",
                    line=dict(color="#1d4ed8", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_machine["timestamp"],
                    y=future_machine["risk_probability"],
                    name="Futur brut",
                    line=dict(color="#fca5a5", width=1, dash="dot"),
                    opacity=0.55,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_machine["timestamp"],
                    y=future_machine["risk_smoothed"],
                    name="Futur lisse",
                    line=dict(color=COLOR_RISK, width=3, dash="dash"),
                )
            )
            fig.add_hline(y=risk_threshold, line_dash="dash", line_color=COLOR_WARN, annotation_text="Seuil")
            if last_history_ts is not None:
                fig.add_shape(
                    type="line",
                    x0=last_history_ts,
                    x1=last_history_ts,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color=COLOR_MUTED, dash="dot", width=2),
                )
                fig.add_annotation(
                    x=last_history_ts,
                    y=1,
                    xref="x",
                    yref="paper",
                    text="Derniere mesure",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(color=COLOR_MUTED),
                )
            fig.update_layout(yaxis_tickformat=".0%", height=390, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Lecture : bleu = historique mesure. Rouge pointille = projection future. On declenche seulement si le futur lisse reste au-dessus du seuil assez longtemps."
            )

            future_alert_rows = future_machine[future_machine["persistent_alert"]]
            if future_alert_rows.empty:
                st.info("Cette machine n'a pas d'alerte persistante dans l'horizon choisi.")
            else:
                alert_preview = future_alert_rows[
                    [
                        col
                        for col in [
                            "timestamp",
                            "forecast_horizon_hours",
                            "risk_smoothed",
                            "risk_probability",
                            "temperature_motor",
                            "vibration_rms",
                            "hours_since_maintenance",
                        ]
                        if col in future_alert_rows.columns
                    ]
                ].copy()
                for col in ["risk_smoothed", "risk_probability"]:
                    if col in alert_preview.columns:
                        alert_preview[col] = (alert_preview[col] * 100).round(1)
                if "forecast_horizon_hours" in alert_preview.columns:
                    alert_preview["forecast_horizon_hours"] = alert_preview["forecast_horizon_hours"].round(2)
                st.dataframe(alert_preview.head(30), width="stretch", hide_index=True)

        st.caption(
            "Important : les lignes futures ne sont pas des capteurs reels. C'est un scenario court terme base sur l'etat recent du dataset, utile pour prioriser, pas pour remplacer une mesure terrain."
        )

with model_tab:
    st.subheader("Performance orientee maintenance")
    if METRICS_CSV_PATH.exists():
        metrics = cached_metrics(str(METRICS_CSV_PATH))
        best_row = metrics.sort_values("f2", ascending=False).iloc[0]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Modele retenu", str(best_row["model"]))
        m2.metric("Recall", pct(float(best_row["recall"])))
        m3.metric("F2-score", pct(float(best_row["f2"])))
        m4.metric("PR-AUC", pct(float(best_row["pr_auc"])))
        m5.metric("Pannes ratees", pct(float(best_row["false_negative_rate"])))

        st.markdown(
            "<div class='section-note'>Le F2-score est priorise car rater une panne coute plus cher qu'une alerte preventive en trop.</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Ces metriques evaluent le modele brut point par point. Le dashboard ajoute ensuite une regle metier de lissage pour reduire les fausses alertes dues aux pics isoles."
        )

        c1, c2 = st.columns([1.15, 1])
        with c1:
            metric_cols = ["f2", "recall", "pr_auc", "balanced_accuracy", "precision", "false_negative_rate"]
            display_metrics = metrics[["model", *metric_cols]].melt("model", var_name="metrique", value_name="score")
            fig = px.bar(
                display_metrics,
                x="model",
                y="score",
                color="metrique",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_layout(height=390, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Lecture : le recall et le F2 sont prioritaires car le projet cherche surtout a eviter les pannes ratees. La precision mesure le cout des fausses alertes."
            )
        with c2:
            st.plotly_chart(plot_confusion_matrix(best_row), width="stretch")
            st.caption(
                "Lecture : la case FN correspond aux pannes ratees. C'est la case la plus importante a reduire en maintenance predictive."
            )

        st.subheader("Variables qui apportent le plus de signal")
        if FEATURE_IMPORTANCE_PATH.exists():
            importance = cached_importance(str(FEATURE_IMPORTANCE_PATH)).head(15)
            fig = px.bar(
                importance.sort_values("importance_mean"),
                x="importance_mean",
                y="feature",
                orientation="h",
                labels={"importance_mean": "Perte de F2 quand la variable est melangee", "feature": "Variable"},
                color_discrete_sequence=["#1d4ed8"],
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Utilite : identifier les variables qui changent vraiment la performance du modele. Plus la barre est grande, plus la variable apporte de l'information."
            )
    else:
        st.warning("Aucun rapport de metriques trouve. Lance python -m src.train")

with data_tab:
    st.subheader("Qualite des donnees")
    report_path = REPORTS_DIR / "data_quality_report.csv"
    if report_path.exists():
        quality = pd.read_csv(report_path, index_col=0)
        q1, q2, q3 = st.columns(3)
        q1.metric("Colonnes avec valeurs manquantes", int((quality["missing_rate"] > 0).sum()))
        q2.metric("Max missing rate", pct(float(quality["missing_rate"].max())))
        q3.metric("Max outlier rate IQR", pct(float(quality["outlier_rate_iqr"].max())))

        c1, c2 = st.columns(2)
        with c1:
            missing = quality["missing_rate"].sort_values(ascending=False).head(10).reset_index()
            missing.columns = ["variable", "missing_rate"]
            fig = px.bar(missing, x="missing_rate", y="variable", orientation="h")
            fig.update_layout(height=350, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Utilite : reperer les capteurs incomplets. Ces valeurs sont imputees dans le pipeline, mais trop de manquants peut rendre un capteur moins fiable."
            )
        with c2:
            outliers = quality["outlier_rate_iqr"].sort_values(ascending=False).head(10).reset_index()
            outliers.columns = ["variable", "outlier_rate"]
            fig = px.bar(outliers, x="outlier_rate", y="variable", orientation="h", color_discrete_sequence=[COLOR_WARN])
            fig.update_layout(height=350, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Utilite : detecter les capteurs avec beaucoup de valeurs extremes. Le modele applique un clipping IQR pour limiter leur impact."
            )

        st.dataframe(quality, width="stretch")
    else:
        st.info("Lance python scripts/analyze_dataset.py pour generer le rapport.")

with simulation_tab:
    st.subheader("Simulation d'un scenario")
    st.markdown(
        "<div class='section-note'>La simulation part du dernier etat reel d'une machine. Elle calcule le risque instantane puis projette un risque futur en tenant compte du mode d'exploitation.</div>",
        unsafe_allow_html=True,
    )

    machine_options = latest_df["machine_id"].dropna().tolist() if "machine_id" in latest_df.columns else []
    selected_machine = st.selectbox("Machine de reference", machine_options, index=0 if machine_options else None, key="sim_machine")

    if selected_machine is not None:
        base_row = latest_df[latest_df["machine_id"] == selected_machine].iloc[0].copy()
        payload = {col: base_row[col] for col in artifact["feature_columns"]}

        adjustable_features = [
            col
            for col in ["vibration_rms", "temperature_motor", "current_phase_avg", "pressure_level", "rpm", "hours_since_maintenance"]
            if col in payload
        ]

        cols = st.columns(3)
        for idx, feature in enumerate(adjustable_features):
            series = pd.to_numeric(scored_df[feature], errors="coerce")
            min_value = float(series.quantile(0.01))
            max_value = float(series.quantile(0.99))
            current_value = float(base_row[feature]) if pd.notna(base_row[feature]) else float(series.median())
            with cols[idx % 3]:
                payload[feature] = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=min(max(current_value, min_value), max_value),
                )

        if "operating_mode" in payload:
            modes = sorted(scored_df["operating_mode"].dropna().astype(str).unique().tolist())
            current_mode = str(base_row["operating_mode"])
            payload["operating_mode"] = st.selectbox(
                "operating_mode",
                modes,
                index=modes.index(current_mode) if current_mode in modes else 0,
            )

        sim_horizon_hours = st.slider(
            "Duree du scenario",
            1,
            TARGET_HORIZON_HOURS,
            TARGET_HORIZON_HOURS,
            1,
            help="Limite a 24h car le modele predit failure_within_24h. Pour 48h ou 72h, il faut entrainer une cible adaptee.",
        )

        if st.button("Calculer le risque", type="primary"):
            prediction = predict_one(payload, artifact)
            risk = float(prediction.get("probability_failure", prediction["prediction"]))
            label = risk_label(risk, risk_threshold)
            css_class = risk_class(risk, risk_threshold)
            st.markdown(
                f"<span class='status-pill {css_class}'>{label} - risque instantane {risk * 100:.1f}%</span>",
                unsafe_allow_html=True,
            )
            st.caption("Risque instantane : meme etat capteur, meme lags, seul le mode selectionne est passe au modele.")

            if "operating_mode" in payload:
                simulated_history = build_simulated_history(
                    df,
                    selected_machine,
                    payload,
                    adjustable_features,
                )
                if simulated_history.empty:
                    st.warning("Projection future impossible pour cette machine: historique timestamp introuvable.")
                else:
                    future_sim = forecast_future_failures(
                        simulated_history,
                        artifact,
                        horizon_hours=sim_horizon_hours,
                        step_minutes=int(round(sampling_minutes or 15)),
                        trend_window=12,
                        scenario="stable_recent",
                        future_mode=str(payload["operating_mode"]),
                    )
                    future_sim = add_persistent_risk_alerts(
                        future_sim,
                        threshold=risk_threshold,
                        smoothing_window=smoothing_window,
                        persistence_points=persistence_points,
                    )

                    future_max_risk = float(future_sim["risk_smoothed"].max()) if len(future_sim) else risk
                    future_end_risk = float(future_sim["risk_smoothed"].iloc[-1]) if len(future_sim) else risk
                    future_alert = bool(future_sim["persistent_alert"].any()) if len(future_sim) else False
                    mode_drift_factor = float(future_sim["mode_drift_factor"].iloc[-1]) if "mode_drift_factor" in future_sim else 1.0
                    projected_hours = (
                        float(future_sim["forecast_elapsed_hours"].iloc[-1])
                        if "forecast_elapsed_hours" in future_sim
                        else float(sim_horizon_hours)
                    )

                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Mode simule", str(payload["operating_mode"]))
                    s2.metric("Facteur capteurs", f"x{mode_drift_factor:.2f}")
                    s3.metric("Duree projetee", f"{projected_hours:.1f} h")
                    s4.metric("Risque futur max", pct(future_max_risk))

                    future_label = "Critique" if future_alert else risk_label(future_max_risk, risk_threshold)
                    future_class = "pill-risk" if future_alert else risk_class(future_max_risk, risk_threshold)
                    st.markdown(
                        f"<span class='status-pill {future_class}'>{future_label} - risque futur fin scenario {future_end_risk * 100:.1f}%</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Projection mode-aware : la duree reste la meme. Idle garde les capteurs quasi stables, normal applique la derive moyenne, peak accelere les deltas capteurs estimes depuis le dataset."
                    )

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=future_sim["timestamp"],
                            y=future_sim["risk_smoothed"],
                            name="Risque futur lisse",
                            line=dict(color=COLOR_RISK, width=3),
                        )
                    )
                    fig.add_hline(y=risk_threshold, line_dash="dash", line_color=COLOR_WARN, annotation_text="Seuil")
                    fig.update_layout(yaxis_tickformat=".0%", height=280, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig, width="stretch")

            with st.expander("Details prediction instantanee"):
                st.json(prediction)
