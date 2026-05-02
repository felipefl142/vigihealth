"""Model registry for VigiHealth.

Each entry returns an Optuna-suggesting factory that yields a fresh estimator
per trial. Tree-based models (LightGBM, XGBoost) tolerate NaN; the linear and
imbalanced-RF candidates need an imputer/scaler chained in front, which the
trainer adds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import optuna
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[optuna.Trial], object]
    needs_scaling: bool
    handles_nan: bool


def _logreg(trial: optuna.Trial) -> LogisticRegression:
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 1e2, log=True),
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        n_jobs=-1,
    )


def _balanced_rf(trial: optuna.Trial) -> BalancedRandomForestClassifier:
    return BalancedRandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_int("max_depth", 4, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        sampling_strategy="auto",
        replacement=True,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )


def _lightgbm(trial: optuna.Trial) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 5e-3, 3e-1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=1,
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        class_weight="balanced",
        verbose=-1,
        random_state=42,
        n_jobs=-1,
    )


def _xgboost(trial: optuna.Trial) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 5e-3, 3e-1, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


_REGISTRY: list[ModelSpec] = [
    ModelSpec("LogisticRegression", _logreg, needs_scaling=True, handles_nan=False),
    ModelSpec("BalancedRandomForest", _balanced_rf, needs_scaling=False, handles_nan=False),
    ModelSpec("LightGBM", _lightgbm, needs_scaling=False, handles_nan=True),
    ModelSpec("XGBoost", _xgboost, needs_scaling=False, handles_nan=True),
]


def get_batch_models(*, skip_logreg: bool = False, skip_boosting: bool = False) -> dict[str, ModelSpec]:
    out: dict[str, ModelSpec] = {}
    for spec in _REGISTRY:
        if skip_logreg and spec.name == "LogisticRegression":
            continue
        if skip_boosting and spec.name in {"LightGBM", "XGBoost"}:
            continue
        out[spec.name] = spec
    return out
