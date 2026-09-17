"""
Stage 2: Model Engineering
- feature engineering (OneHot + Scale)
- tune CatBoost, XGBoost, LightGBM with Optuna
- log to MLflow (SQLite backend)
- save best pipeline to models/best_model.joblib
"""
import os
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_EXPERIMENT = "adult_income"

N_TRIALS = 5

NUMERIC_COLS = ["age", "fnlwgt", "education-num",
                "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_COLS = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]
TARGET = "income"


def setup_mlflow():
    """Настраивает MLflow на SQLite-бэкенд (файловый больше не поддерживается)."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    log.info(f"MLflow tracking URI: {tracking_uri}")


def load_data():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    return X_train, y_train, X_test, y_test


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def suggest_params(model_name, trial):
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
    raise ValueError(f"Unknown model: {model_name}")


def build_model(model_name, params):
    if model_name == "catboost":
        return CatBoostClassifier(verbose=0, random_state=42, **params)
    if model_name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **params,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
    raise ValueError(f"Unknown model: {model_name}")


def tune_model(model_name, X_train, y_train, X_test, y_test):
    log.info(f"=== Tuning {model_name} ===")

    def objective(trial):
        params = suggest_params(model_name, trial)
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", build_model(model_name, params)),
        ])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, proba)

    study = optuna.create_study(direction="maximize",
                                study_name=f"{model_name}_study")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_auc = study.best_value
    log.info(f"{model_name} best AUC={best_auc:.4f} params={best_params}")

    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", build_model(model_name, best_params)),
    ])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    return pipe, metrics, best_params


def main():
    setup_mlflow()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()
    log.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = {}
    best_score = -1.0
    best_pipeline = None
    best_name = None
    best_metrics = None
    SKOPS_TRUSTED = [
        "catboost.core.CatBoostClassifier",
        "xgboost.core.XGBClassifier",
        "lightgbm.sklearn.LGBMClassifier",
    ]

    for name in ["catboost", "xgboost", "lightgbm"]:
        with mlflow.start_run(run_name=name):
            pipe, metrics, params = tune_model(name, X_train, y_train, X_test, y_test)

            mlflow.log_params({f"best_{k}": v for k, v in params.items()})
            mlflow.log_metrics(metrics)

            try:
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    name="model",
                    skops_trusted_types=SKOPS_TRUSTED,
                )
            except Exception as e:
                log.warning(f"Could not log {name} to MLflow: {e}")

            results[name] = {"metrics": metrics, "params": params}
            log.info(f"{name} -> {metrics}")

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_pipeline = pipe
                best_name = name
                best_metrics = metrics

    out_path = MODELS_DIR / "best_model.joblib"
    joblib.dump({
        "pipeline": best_pipeline,
        "model_name": best_name,
        "metrics": best_metrics,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }, out_path)
    log.info(f"Best model: {best_name} (AUC={best_score:.4f}) saved to {out_path}")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"best": best_name, "all": results}, f, indent=2)


if __name__ == "__main__":
    main()