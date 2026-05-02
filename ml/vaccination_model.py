"""Vaccination milestone model training.

Trains a single classifier across all (vaccine, horizon) combinations from
``data/gold/abt_vaccination_milestone.parquet``. Vaccine and horizon enter the
feature matrix as one-hot categoricals so the model can leverage cross-vaccine
structure.

Training protocol:
  * Time-based split: train year <= ``split_year`` (default 2018), test after.
  * Inner Optuna search uses the last two years of train as validation.
  * Primary metric: PR-AUC. Also logged: ROC-AUC, F1, Brier, confusion matrix,
    feature importance, holdout predictions CSV.
  * MLflow tracking URI is sqlite:///mlflow.db.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from constants import GOLD_DIR, MLFLOW_DB, MLRUNS_DIR
from ml.model_selection import ModelSpec, get_batch_models

logger = logging.getLogger(__name__)

ABT_PATH = GOLD_DIR / "abt_vaccination_milestone.parquet"
EXPERIMENT_NAME = "vaccination_milestone"
ID_COLS = ("country_iso3", "year", "label")
CATEGORICAL_COLS = ("vaccine", "horizon")

# Names referenced by tests/test_api_contract.py.
VACCINATION_FEATURES = ["vaccine", "horizon"]


def _setup_mlflow() -> None:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def _load_abt() -> pd.DataFrame:
    if not ABT_PATH.exists():
        raise FileNotFoundError(f"{ABT_PATH} missing. Run etl.gold first.")
    return pd.read_parquet(ABT_PATH)


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    return df[feature_cols].copy(), df["label"].astype("int8")


def _build_preprocessor(X: pd.DataFrame, *, scale_numeric: bool, impute_numeric: bool) -> ColumnTransformer:
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    num_steps = []
    if impute_numeric:
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe: object = Pipeline(num_steps) if num_steps else "passthrough"

    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def _make_pipeline(spec: ModelSpec, X: pd.DataFrame, trial: optuna.Trial) -> Pipeline:
    pre = _build_preprocessor(
        X,
        scale_numeric=spec.needs_scaling,
        impute_numeric=not spec.handles_nan,
    )
    estimator = spec.factory(trial)
    return Pipeline([("pre", pre), ("clf", estimator)])


def _time_split(df: pd.DataFrame, split_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= split_year].reset_index(drop=True)
    test = df[df["year"] > split_year].reset_index(drop=True)
    return train, test


def _inner_validation(train: pd.DataFrame, val_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = train["year"].max() - val_years
    inner_train = train[train["year"] <= cutoff].reset_index(drop=True)
    inner_val = train[train["year"] > cutoff].reset_index(drop=True)
    return inner_train, inner_val


def _objective_factory(spec: ModelSpec, train: pd.DataFrame, val_years: int):
    inner_train, inner_val = _inner_validation(train, val_years)
    X_tr, y_tr = _split_xy(inner_train)
    X_vl, y_vl = _split_xy(inner_val)

    def objective(trial: optuna.Trial) -> float:
        pipe = _make_pipeline(spec, X_tr, trial)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_vl)[:, 1]
        return float(average_precision_score(y_vl, proba))

    return objective


def _final_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype("int8")
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def _feature_importance(pipe: Pipeline) -> pd.DataFrame | None:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        return None

    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_).ravel()
    else:
        return None

    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _log_artifacts(
    pipe: Pipeline,
    test_df: pd.DataFrame,
    proba: np.ndarray,
    metrics: dict[str, float],
    model_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Holdout predictions
        preds = test_df[["country_iso3", "year", "vaccine", "horizon", "label"]].copy()
        preds["proba"] = proba
        preds["pred"] = (proba >= 0.5).astype("int8")
        preds_path = tmp_path / "holdout_predictions.csv"
        preds.to_csv(preds_path, index=False)
        mlflow.log_artifact(str(preds_path))

        # Confusion matrix per (vaccine, horizon)
        cm_rows = []
        for (v, h), grp in preds.groupby(["vaccine", "horizon"]):
            cm = confusion_matrix(grp["label"], grp["pred"], labels=[0, 1])
            cm_rows.append({
                "vaccine": v, "horizon": h,
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
                "n": len(grp),
            })
        cm_df = pd.DataFrame(cm_rows)
        cm_path = tmp_path / "confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=False)
        mlflow.log_artifact(str(cm_path))

        # Feature importance plot (top 25)
        fi = _feature_importance(pipe)
        if fi is not None and len(fi) > 0:
            top = fi.head(25)
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh(top["feature"][::-1], top["importance"][::-1])
            ax.set_title(f"{model_name} top 25 features")
            ax.set_xlabel("importance")
            fig.tight_layout()
            fi_plot = tmp_path / "feature_importance.png"
            fig.savefig(fi_plot, dpi=120)
            plt.close(fig)
            mlflow.log_artifact(str(fi_plot))

            fi_csv = tmp_path / "feature_importance.csv"
            fi.to_csv(fi_csv, index=False)
            mlflow.log_artifact(str(fi_csv))


def _train_one_model(
    spec: ModelSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_trials: int,
    val_years: int,
    split_year: int,
) -> dict[str, float]:
    logger.info("=== %s — Optuna study (%d trials) ===", spec.name, n_trials)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        study_name=f"{EXPERIMENT_NAME}_{spec.name}",
    )
    study.optimize(_objective_factory(spec, train, val_years), n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_inner_pr_auc = study.best_value
    logger.info("%s best inner PR-AUC=%.4f params=%s", spec.name, best_inner_pr_auc, best_params)

    # Refit best on full train
    best_trial = optuna.trial.FixedTrial(best_params)
    X_train, y_train = _split_xy(train)
    X_test, y_test = _split_xy(test)
    final_pipe = _make_pipeline(spec, X_train, best_trial)
    final_pipe.fit(X_train, y_train)
    proba = final_pipe.predict_proba(X_test)[:, 1]
    metrics = _final_metrics(y_test.to_numpy(), proba)
    logger.info("%s holdout: %s", spec.name, metrics)

    with mlflow.start_run(run_name=spec.name):
        mlflow.log_params(best_params)
        mlflow.log_param("model", spec.name)
        mlflow.log_param("split_year", split_year)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("val_years", val_years)
        mlflow.log_param("n_train", len(train))
        mlflow.log_param("n_test", len(test))
        mlflow.log_metric("inner_pr_auc", best_inner_pr_auc)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(final_pipe, name="model")
        _log_artifacts(final_pipe, test, proba, metrics, spec.name)

    return metrics


def train_vaccination_models(
    *,
    n_trials: int = 30,
    split_year: int = 2018,
    val_years: int = 2,
    skip_logreg: bool = False,
    skip_boosting: bool = False,
) -> dict[str, dict[str, float]]:
    _setup_mlflow()
    df = _load_abt()
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    train, test = _time_split(df, split_year)
    if test.empty:
        raise ValueError(f"Empty test set for split_year={split_year}")
    logger.info("ABT: %d rows. train=%d (years <=%d), test=%d (years >%d)",
                len(df), len(train), split_year, len(test), split_year)

    specs = get_batch_models(skip_logreg=skip_logreg, skip_boosting=skip_boosting)
    results: dict[str, dict[str, float]] = {}
    for name, spec in specs.items():
        results[name] = _train_one_model(
            spec, train, test,
            n_trials=n_trials, val_years=val_years, split_year=split_year,
        )

    summary = pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
    logger.info("\n=== model comparison (sorted by holdout PR-AUC) ===\n%s",
                summary.to_string())
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train vaccination milestone models.")
    p.add_argument("--trials", type=int, default=30, help="Optuna trials per model.")
    p.add_argument("--split-year", type=int, default=2018, help="Train years <= this; test years > this.")
    p.add_argument("--val-years", type=int, default=2, help="Last N train years used as inner validation set.")
    p.add_argument("--nologreg", action="store_true", help="Skip LogisticRegression.")
    p.add_argument("--no-boosting", action="store_true", help="Skip LightGBM and XGBoost.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_vaccination_models(
        n_trials=args.trials,
        split_year=args.split_year,
        val_years=args.val_years,
        skip_logreg=args.nologreg,
        skip_boosting=args.no_boosting,
    )


if __name__ == "__main__":
    main()
