"""Train and evaluate a baseline failure-prediction model (Logistic Regression)."""

import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "predictive_maintenance.db"
RANDOM_STATE = 42

# Columns that must never be used as features:
# identifiers carry no physical meaning, and the failure-mode flags define the target (leakage).
ID_COLUMNS = ["udi", "product_id"]
LEAKY_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]

NUMERIC_FEATURES = [
    "air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min",
    "temp_diff_k", "power_w", "wear_torque",
]
CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "machine_failure"


def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise SystemExit("Database not found. Run: python src/load_to_sqlite.py")
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM machine_data", conn)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-motivated features built only from sensor readings."""
    df = df.copy()
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    # Power (W) = torque (Nm) x angular speed (rad/s); rpm * 2*pi/60 = rad/s
    df["power_w"] = df["torque_nm"] * df["rot_speed_rpm"] * 3.141592653589793 * 2 / 60
    df["wear_torque"] = df["tool_wear_min"] * df["torque_nm"]
    return df


def build_logistic_regression() -> Pipeline:
    # The scaler and encoder are fitted on training data only (inside the pipeline),
    # so no information from the test set leaks into preprocessing.
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    return Pipeline([("prep", preprocess), ("model", model)])

def build_random_forest() -> Pipeline:
    # Trees do not need scaling, so numeric columns pass through unchanged.
    preprocess = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,  # my choice to limit overfitting, not tuned
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", preprocess), ("model", model)])


def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)  # default decision threshold: 0.5

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    results = {
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "PR-AUC": average_precision_score(y_test, proba),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }

    print(f"\n=== {name} ===")
    print(f"Confusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    for metric, value in results.items():
        print(f"{metric:10s}: {value:.3f}")
    print(f"(Accuracy {accuracy_score(y_test, pred):.3f} is shown for reference only; "
          "do not rely on it for imbalanced data.)")
    return results


def main() -> None:
    df = add_features(load_data())

    # Safety check: no identifier or failure-mode column may enter the features
    forbidden = set(ID_COLUMNS + LEAKY_COLUMNS) & set(FEATURES)
    assert not forbidden, f"Leaky or ID columns in features: {forbidden}"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows, failure rate {y_train.mean():.4f}")
    print(f"Test:  {len(X_test)} rows, failure rate {y_test.mean():.4f} "
          f"(PR-AUC of a random model is about this value)")

    models = {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest": build_random_forest(),
    }

    all_results = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        all_results[name] = evaluate(f"{name} (class_weight='balanced')", pipeline, X_test, y_test)

    print("\n=== Summary (paste into README exactly as printed) ===")
    print("| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |")
    print("|---|---|---|---|---|---|")
    for name, r in all_results.items():
        print(f"| {name} | {r['Precision']:.3f} | {r['Recall']:.3f} | {r['F1']:.3f} | "
              f"{r['PR-AUC']:.3f} | {r['ROC-AUC']:.3f} |")

    # Which inputs matter most? Shuffle one column at a time on the held-out test set
    # and measure how much PR-AUC drops.
    rf = models["Random Forest"]
    perm = permutation_importance(
        rf, X_test, y_test, scoring="average_precision",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance = (
        pd.DataFrame({
            "feature": X_test.columns,
            "mean_drop": perm.importances_mean,
            "std": perm.importances_std,
        })
        .sort_values("mean_drop", ascending=False)
    )
    print("\n=== Random Forest: permutation importance (drop in PR-AUC when shuffled) ===")
    print(importance.round(4).to_string(index=False))


if __name__ == "__main__":
    main()