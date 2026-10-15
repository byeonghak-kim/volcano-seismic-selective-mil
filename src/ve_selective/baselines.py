"""Validation-selected classical baselines for the 144-feature representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class BaselineFit:
    models: dict[str, object]
    selected_parameters: dict[str, dict]
    validation_search: pd.DataFrame


def _score(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    per_class = f1_score(y, prediction, labels=np.arange(3), average=None, zero_division=0)
    return (
        float(f1_score(y, prediction, average="macro", zero_division=0)),
        float(accuracy_score(y, prediction)),
        float(per_class.min()),
    )


def fit_feature_baselines(
    features: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    seed: int = 42,
    quick: bool = False,
) -> BaselineFit:
    """Fit LR, RF, RBF-SVM, and HGB; select hyperparameters on validation only."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    selected: dict[str, object] = {}
    parameters: dict[str, dict] = {}
    rows: list[dict] = []

    grids = {
        "Logistic regression": [("C", value) for value in ((0.1, 1.0) if quick else (0.01, 0.1, 1.0, 10.0))],
        "Random forest": [
            ("settings", (depth, leaf))
            for depth in ((12,) if quick else (None, 12, 24))
            for leaf in ((3,) if quick else (1, 3, 8))
        ],
        "RBF-SVM": [
            ("settings", (c, gamma))
            for c in ((1.0,) if quick else (0.1, 1.0, 10.0, 100.0))
            for gamma in (("scale",) if quick else ("scale", 0.01, 0.001))
        ],
        "Histogram gradient boosting": [
            ("settings", (rate, leaves, l2))
            for rate in ((0.1,) if quick else (0.03, 0.1))
            for leaves in ((15,) if quick else (15, 31, 63))
            for l2 in ((1.0,) if quick else (0.0, 1.0))
        ],
    }
    for name, candidates in grids.items():
        fitted = []
        for _, setting in candidates:
            if name == "Logistic regression":
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", LogisticRegression(C=setting, class_weight="balanced", max_iter=5000, random_state=seed)),
                    ]
                )
                metadata = {"C": setting}
            elif name == "Random forest":
                depth, leaf = setting
                model = RandomForestClassifier(
                    n_estimators=40 if quick else 400,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=1 if quick else -1,
                )
                metadata = {"max_depth": depth, "min_samples_leaf": leaf}
            elif name == "RBF-SVM":
                c, gamma = setting
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", SVC(C=c, gamma=gamma, kernel="rbf", class_weight="balanced", decision_function_shape="ovr", random_state=seed)),
                    ]
                )
                metadata = {"C": c, "gamma": gamma}
            else:
                rate, leaves, l2 = setting
                model = HistGradientBoostingClassifier(
                    learning_rate=rate,
                    max_leaf_nodes=leaves,
                    l2_regularization=l2,
                    max_iter=40 if quick else 300,
                    class_weight="balanced",
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=10 if quick else 25,
                    random_state=seed,
                )
                metadata = {"learning_rate": rate, "max_leaf_nodes": leaves, "l2_regularization": l2}
            model.fit(x[train_index], y[train_index])
            score = _score(y[validation_index], model.predict(x[validation_index]))
            rows.append({"model": name, **metadata, "validation_macro_f1": score[0], "validation_accuracy": score[1], "minimum_class_f1": score[2]})
            fitted.append((score, metadata, model))
        best_score, best_metadata, best_model = max(fitted, key=lambda item: item[0])
        selected[name] = best_model
        parameters[name] = {**best_metadata, "validation_score": best_score}
    return BaselineFit(selected, parameters, pd.DataFrame(rows))


def baseline_logits(model: object, features: np.ndarray) -> np.ndarray:
    """Return score-like logits suitable for validation temperature fitting."""
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(features), dtype=np.float64)
        return score if score.ndim == 2 else np.column_stack([-score, score])
    probability = np.asarray(model.predict_proba(features), dtype=np.float64)
    return np.log(np.clip(probability, 1e-12, 1.0))
