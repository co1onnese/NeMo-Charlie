"""Reusable metrics computation for evaluation outputs."""

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


@dataclass
class ClassificationMetrics:
    accuracy: float
    samples: int
    report: Dict[str, Dict[str, float]]


@dataclass
class FinancialMetrics:
    hit_rate: float
    mean_return: float
    std_return: float
    sharpe_ratio: float
    samples: int


def compute_classification_metrics(df: pd.DataFrame, labels: Iterable[str]) -> ClassificationMetrics:
    mask = df["gt_action"].notna() & df["pred_action"].notna()
    if mask.sum() == 0:
        return ClassificationMetrics(accuracy=0.0, samples=0, report={})

    y_true = df.loc[mask, "gt_action"].tolist()
    y_pred = df.loc[mask, "pred_action"].tolist()

    report = classification_report(y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0)
    accuracy = report.get("accuracy", 0.0)
    return ClassificationMetrics(accuracy=accuracy, samples=mask.sum(), report=report)


def compute_financial_metrics(df: pd.DataFrame, action_directions: Dict[str, int]) -> FinancialMetrics:
    valid = df.dropna(subset=["realized_return", "pred_action"])
    valid = valid[valid["pred_action"].isin(action_directions)]
    if len(valid) == 0:
        return FinancialMetrics(0.0, 0.0, float("nan"), float("nan"), 0)

    def direction_correct(row):
        direction = action_directions.get(row["pred_action"], 0)
        if direction == 0:
            return abs(row["realized_return"]) < 0.001
        return np.sign(row["realized_return"]) == direction

    valid = valid.assign(direction_correct=valid.apply(direction_correct, axis=1))
    hit_rate = valid["direction_correct"].mean()
    mean_return = valid["realized_return"].mean()
    std_return = valid["realized_return"].std()
    sharpe = float("nan")
    if std_return and std_return > 0:
        sharpe = (mean_return / std_return) * np.sqrt(252)

    return FinancialMetrics(
        hit_rate=hit_rate,
        mean_return=mean_return,
        std_return=std_return,
        sharpe_ratio=sharpe,
        samples=len(valid),
    )

