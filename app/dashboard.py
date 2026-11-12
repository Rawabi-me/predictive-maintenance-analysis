"""Streamlit dashboard for the AI4I 2020 predictive maintenance project."""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "predictive_maintenance.db"
WEAR_LABELS = ["0-49", "50-99", "100-149", "150-199", "200+"]
MIN_FAILURES_FOR_RATES = 30

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM machine_data", conn)
    df["wear_range"] = pd.cut(
        df["tool_wear_min"],
        bins=[0, 50, 100, 150, 200, np.inf],
        right=False,
        labels=WEAR_LABELS,
    )
    return df


def failure_rate_bar(data: pd.DataFrame, group_col: str, xlabel: str) -> None:
    summary = data.groupby(group_col, observed=True)["machine_failure"].agg(
        records="count", failure_rate="mean"
    )
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(summary.index.astype(str), summary["failure_rate"] * 100)
    for bar, n in zip(bars, summary["records"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Failure rate (%)")
    ax.margins(y=0.15)
    st.pyplot(fig)
    plt.close(fig)


def speed_torque_scatter(data: pd.DataFrame) -> None:
    normal = data[data["machine_failure"] == 0]
    failed = data[data["machine_failure"] == 1]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(normal["rot_speed_rpm"], normal["torque_nm"],
               s=4, alpha=0.3, label="No failure")
    ax.scatter(failed["rot_speed_rpm"], failed["torque_nm"],
               s=10, alpha=0.8, color="tab:red", label="Failure")
    ax.set_xlabel("Rotational speed (rpm)")
    ax.set_ylabel("Torque (Nm)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.title("Predictive Maintenance: Machine Failure Dashboard")
    st.caption(
        "Synthetic dataset (AI4I 2020, UCI Machine Learning Repository). "
        "It is not real operational data."
    )

    if not DB_PATH.exists():
        st.error("Database not found. Run `python src/load_to_sqlite.py` first.")
        st.stop()

    df = load_data()

    st.sidebar.header("Filters")
    all_types = sorted(df["type"].unique())
    types = st.sidebar.multiselect("Product type", all_types, default=all_types)
    wear = st.sidebar.multiselect("Tool wear range (min)", WEAR_LABELS, default=WEAR_LABELS)

    filtered = df[df["type"].isin(types) & df["wear_range"].isin(wear)]
    if filtered.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(filtered):,}")
    c2.metric("Failure rate", f"{filtered['machine_failure'].mean() * 100:.2f}%")
    c3.metric("Avg tool wear (min)", f"{filtered['tool_wear_min'].mean():.1f}")

    if filtered["machine_failure"].sum() < MIN_FAILURES_FOR_RATES:
        st.warning(
            f"Fewer than {MIN_FAILURES_FOR_RATES} failures in this selection: "
            "failure rates are based on very few events and may be unreliable."
        )

    left, middle, right = st.columns(3)
    with left:
        st.subheader("By tool wear")
        failure_rate_bar(filtered, "wear_range", "Tool wear (min)")
    with middle:
        st.subheader("By product type")
        failure_rate_bar(filtered, "type", "Product type")
    with right:
        st.subheader("Speed vs torque")
        speed_torque_scatter(filtered)

    st.caption(
        "Failure rate = share of records with machine_failure = 1. "
        "Patterns are descriptive and come from a synthetic dataset; "
        "they show correlation, not causation."
    )


if __name__ == "__main__":
    main()