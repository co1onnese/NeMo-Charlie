"""Evaluation helpers for action parsing and metrics configuration."""

import json
import re
from pathlib import Path
from typing import Dict, Iterable


ACTION_PATTERN = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE)


def extract_action(text: str) -> str:
    if not text or not isinstance(text, str):
        return "UNKNOWN"
    match = ACTION_PATTERN.search(text)
    if match:
        return match.group(1).strip().upper().replace(" ", "_")

    txt = text.upper()
    if "STRONG BUY" in txt or "STRONG_BUY" in txt:
        return "STRONG_BUY"
    if "BUY" in txt:
        return "BUY"
    if "HOLD" in txt:
        return "HOLD"
    if "SELL" in txt:
        return "SELL"
    if "STRONG SELL" in txt or "STRONG_SELL" in txt:
        return "STRONG_SELL"
    return "UNKNOWN"


def load_metrics_config(path: Path = Path("configs/eval_config.json")) -> "MetricsConfig":
    if not path.is_file():
        return MetricsConfig.default()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return MetricsConfig.from_dict(data)


class MetricsConfig:
    def __init__(self, classification: Iterable[str], financial: Dict[str, int]):
        self.classification = tuple(classification)
        self.financial = dict(financial)

    @classmethod
    def from_dict(cls, data: Dict) -> "MetricsConfig":
        default_financial = {
            "STRONG_BUY": 1,
            "BUY": 1,
            "HOLD": 0,
            "SELL": -1,
            "STRONG_SELL": -1,
        }
        labels = data.get("classification_labels", list(default_financial.keys()))
        direction = data.get("financial_direction", default_financial)
        return cls(labels, direction)

    @classmethod
    def default(cls) -> "MetricsConfig":
        return cls.from_dict({})

