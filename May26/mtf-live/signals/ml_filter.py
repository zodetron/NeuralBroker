"""
signals/ml_filter.py — XGBoost ML filter for live signals.

Two separate models:
  mtf_model    — filters MTF_LONG / MTF_SHORT signals
  swappy_model — filters SWAPPY_LONG / SWAPPY_SHORT signals

Training:
  Sliding window (8 weeks train / 1 week test) against DuckDB trade history.
  Label: 1 if trade hit TP within TIME_EXIT_BARS × 5M bars, else 0.
  Features are extracted from the signal's tf_stack snapshot.
  Models are re-trained when enough new labeled samples accumulate (≥ 40).
  Saved/loaded from MODELS_DIR/*.joblib.

Usage:
  ml = MLFilter()
  signal = ml.score(signal)   # adds raw_score and approved fields
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import (
    ML_THRESHOLD_STANDARD, ML_THRESHOLD_CONFLUENCE,
    MODELS_DIR,
)

log = logging.getLogger(__name__)

_XGB_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "logloss",
    verbosity        = 0,
    random_state     = 42,
    use_label_encoder = False,
)

_MODEL_FILES = {
    "mtf":    os.path.join(MODELS_DIR, "mtf_model.joblib"),
    "swappy": os.path.join(MODELS_DIR, "swappy_model.joblib"),
}


class MLFilter:
    """
    Scores signals via XGBoost.  Falls back to approve-all if no model is
    trained yet (not enough history).
    """

    def __init__(self) -> None:
        self._models: dict[str, Optional[XGBClassifier]] = {
            "mtf":    self._load("mtf"),
            "swappy": self._load("swappy"),
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, signal: dict) -> dict:
        """
        Annotate signal with raw_score and approved.
        Returns signal dict (mutated in-place for convenience).
        """
        model_key = "swappy" if signal["type"].startswith("SWAPPY") else "mtf"
        model     = self._models.get(model_key)

        if model is None:
            # No model yet — pass all signals through
            signal["raw_score"] = 0.5
            signal["approved"]  = True
            return signal

        feats = _extract_features(signal)
        X     = np.array(feats).reshape(1, -1)

        try:
            prob = float(model.predict_proba(X)[0][1])
        except Exception as exc:
            log.warning("ML score failed: %s — passing signal", exc)
            signal["raw_score"] = 0.5
            signal["approved"]  = True
            return signal

        threshold = ML_THRESHOLD_CONFLUENCE if signal["confluence"] else ML_THRESHOLD_STANDARD

        signal["raw_score"] = round(prob, 4)
        signal["approved"]  = prob >= threshold
        return signal

    def retrain(self, model_key: str, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a new model and persist it to disk."""
        if len(X) < 40:
            log.info("Not enough samples to retrain %s (%d)", model_key, len(X))
            return

        clf = XGBClassifier(**_XGB_PARAMS)
        clf.fit(X, y)

        path = _MODEL_FILES[model_key]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(clf, path)
        self._models[model_key] = clf
        log.info("Retrained %s model on %d samples", model_key, len(X))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load(self, key: str) -> Optional[XGBClassifier]:
        path = _MODEL_FILES[key]
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                log.info("Loaded %s model from %s", key, path)
                return model
            except Exception as exc:
                log.warning("Failed to load %s model: %s", key, exc)
        return None


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_features(signal: dict) -> list[float]:
    """
    Convert a signal's tf_stack snapshots into a fixed-length feature vector.
    Must stay consistent between training and scoring.
    """
    stack = signal.get("tf_stack", {})

    def _g(d: dict, key: str, default: float = 0.0) -> float:
        v = d.get(key, default)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    d1  = stack.get("1d",  {})
    d4h = stack.get("4h",  {})
    d1h = stack.get("1h",  {})
    d15 = stack.get("15m", {})
    d5  = stack.get("5m",  {})

    # 1D features
    bias_1d         = float(_g(d1, "bias"))
    daily_range_pct = _g(d1, "daily_range_pct")

    # 4H features
    zone_map = {"DEMAND": 1, "SUPPLY": -1, "MIDZONE": 0}
    zone_4h  = float(zone_map.get(str(d4h.get("zone", "MIDZONE")), 0))

    # 1H features
    struct_map = {
        "TRENDING_UP": 2, "BREAKOUT_UP": 1,
        "RANGING": 0,
        "TRENDING_DOWN": -2, "BREAKOUT_DOWN": -1,
    }
    struct_1h = float(struct_map.get(str(d1h.get("structure", "RANGING")), 0))
    adx_1h    = _g(d1h, "adx")
    rsi_1h    = _g(d1h, "rsi", 50.0)

    # 15M features
    mom_map = {
        "STRONG_BULL": 2, "PULLBACK_BULL": 1, "NEUTRAL": 0,
        "PULLBACK_BEAR": -1, "STRONG_BEAR": -2, "EXHAUSTED": 0,
    }
    momentum = float(mom_map.get(str(d15.get("momentum", "NEUTRAL")), 0))
    rsi_15m  = _g(d15, "rsi", 50.0)
    adx_15m  = _g(d15, "adx")

    # Signal features
    rr         = _g(signal, "rr")
    atr_norm   = _g(signal, "atr") / max(_g(d1h, "atr", 1.0), 1.0)
    direction  = float(signal.get("direction", 0))
    confluence = float(signal.get("confluence", False))
    is_swappy  = float(signal.get("type", "").startswith("SWAPPY"))

    return [
        bias_1d, daily_range_pct,
        zone_4h,
        struct_1h, adx_1h, rsi_1h,
        momentum, rsi_15m, adx_15m,
        rr, atr_norm, direction, confluence, is_swappy,
    ]
