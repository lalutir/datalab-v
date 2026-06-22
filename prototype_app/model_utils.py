"""
Model utilities for the Jijiga Flood & Drought Risk Prediction prototype.

This module replicates the exact preprocessing, feature engineering, and
prediction logic from:
  - phase3_index_eda.ipynb (risk label construction, normalization params)
  - phase4_feature_engineering.ipynb (lag features)
  - phase5_xgboost.ipynb (XGBoost models)
  - phase6_foundation_model.ipynb (hybrid ensemble: XGBoost + GenCast flood score)

No model retraining is performed. All parameters are derived from the
training period (2001-2022) stored in era5_labeled.parquet.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


# ── Paths (relative to prototype_app/) ────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "src" / "data" / "processed"
XGB_DIR = DATA_DIR / "xgb_models"
ERA5_LABELED_PATH = DATA_DIR / "era5_labeled.parquet"
FEATURE_MATRIX_PATH = DATA_DIR / "feature_matrix.parquet"

# ── Constants (from CLAUDE.md and notebooks) ──────────────────────────────────
FEATURE_VARS = ["spei_6", "api_92", "smi_fc", "total_ro", "tp", "t2m", "e", "pev"]
LAGS = [1, 3, 7, 14, 365]
FEATURE_COLS = [f"{v}_lag_{l}" for v in FEATURE_VARS for l in LAGS]
HORIZONS = [1, 3, 7]
API_DECAY = 0.92
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}
LABEL_MAP_NL = {0: "Laag", 1: "Matig", 2: "Hoog", 3: "Extreem"}


class HybridPredictor:
    """
    Loads XGBoost models and training-period parameters, then produces
    hybrid ensemble predictions combining XGBoost + GenCast-style flood scores.
    """

    def __init__(self):
        self._load_era5_labeled()
        self._compute_training_params()
        self._load_xgboost_models()
        self._load_feature_matrix()

    def _load_era5_labeled(self):
        """Load ERA5 labeled dataset and set time index."""
        self.df = pd.read_parquet(ERA5_LABELED_PATH)
        self.df["time"] = pd.to_datetime(self.df["time"])
        self.df = self.df.set_index("time").sort_index()

    def _compute_training_params(self):
        """
        Compute all normalization parameters from training data (2001-2022).
        Replicates logic from phase3 and postprocess_gencast_results.py.
        """
        train = self.df[self.df.index.year <= 2022]

        # Flood score normalization (min/max from training)
        self.norm_params = {}
        for col in ["api_92", "smi_fc", "total_ro"]:
            self.norm_params[col] = (
                float(train[col].min()),
                float(train[col].max()),
            )

        # Flood score percentile thresholds
        tr_scores = self._calc_flood_scores_array(
            train["api_92"].values,
            train["smi_fc"].values,
            train["total_ro"].values,
        )
        self.flood_p65 = float(np.percentile(tr_scores, 65))
        self.flood_p80 = float(np.percentile(tr_scores, 80))
        self.flood_p90 = float(np.percentile(tr_scores, 90))

        # Drought modifier thresholds
        self.api_p25 = float(train["api_92"].quantile(0.25))
        self.smi_p25 = float(train["smi_fc"].quantile(0.25))

    def _calc_flood_scores_array(self, api_arr, smi_arr, ro_arr):
        """Vectorized flood score computation."""
        mn_a, mx_a = self.norm_params["api_92"]
        mn_s, mx_s = self.norm_params["smi_fc"]
        mn_r, mx_r = self.norm_params["total_ro"]

        n_api = np.clip((api_arr - mn_a) / (mx_a - mn_a + 1e-12), 0, 1)
        n_smi = np.clip((smi_arr - mn_s) / (mx_s - mn_s + 1e-12), 0, 1)
        n_ro = np.clip((ro_arr - mn_r) / (mx_r - mn_r + 1e-12), 0, 1)

        return 0.40 * n_api + 0.35 * n_smi + 0.25 * n_ro

    def _calc_flood_score_scalar(self, api, smi, ro=0.0):
        """Single-value flood score computation."""
        mn_a, mx_a = self.norm_params["api_92"]
        mn_s, mx_s = self.norm_params["smi_fc"]
        mn_r, mx_r = self.norm_params["total_ro"]

        n_api = np.clip((api - mn_a) / (mx_a - mn_a + 1e-12), 0, 1)
        n_smi = np.clip((smi - mn_s) / (mx_s - mn_s + 1e-12), 0, 1)
        n_ro = np.clip((ro - mn_r) / (mx_r - mn_r + 1e-12), 0, 1)

        return 0.40 * n_api + 0.35 * n_smi + 0.25 * n_ro

    def _flood_label(self, score: float) -> int:
        """Classify flood score into risk level (4-class)."""
        if score < self.flood_p65:
            return 0
        elif score < self.flood_p80:
            return 1
        elif score < self.flood_p90:
            return 2
        else:
            return 3

    def _drought_label(self, spei: float, api_val: float, smi_val: float) -> int:
        """
        Apply McKee SPEI thresholds + modifier rule.
        Replicates phase3/postprocess_gencast_results.py logic.
        """
        if spei >= -0.5:
            base = 0
        elif spei >= -1.0:
            base = 1
        elif spei >= -1.5:
            base = 2
        else:
            base = 3

        # Modifier: elevate by 1 if base is 1 or 2 and API or SMI below p25
        if 1 <= base <= 2 and (api_val < self.api_p25 or smi_val < self.smi_p25):
            base = min(base + 1, 3)

        return base

    def _load_xgboost_models(self):
        """Load pre-trained XGBoost models for flood and drought."""
        self.xgb_flood = {}
        self.xgb_drought = {}

        for horizon in HORIZONS:
            # Flood models
            flood_path = XGB_DIR / f"flood_+{horizon}d.ubj"
            if flood_path.exists():
                m = xgb.XGBClassifier()
                m.load_model(str(flood_path))
                self.xgb_flood[horizon] = m

            # Drought models
            drought_path = XGB_DIR / f"drought_+{horizon}d.ubj"
            if drought_path.exists():
                m = xgb.XGBClassifier()
                m.load_model(str(drought_path))
                self.xgb_drought[horizon] = m

    def _load_feature_matrix(self):
        """Load feature matrix for XGBoost predictions."""
        self.fm = pd.read_parquet(FEATURE_MATRIX_PATH)
        self.fm["time"] = pd.to_datetime(self.fm["time"])
        self.fm = self.fm.set_index("time").sort_index()

    def get_last_known_state(self) -> dict:
        """
        Get the last known ERA5 state (most recent row in the dataset).
        Used as initialization for forecast computations.
        """
        last_row = self.df.iloc[-1]
        return {
            "date": self.df.index[-1],
            "api_92": float(last_row["api_92"]),
            "smi_fc": float(last_row["smi_fc"]),
            "spei_6": float(last_row["spei_6"]),
            "total_ro": float(last_row["total_ro"]),
            "tp": float(last_row["tp"]),
            "t2m": float(last_row["t2m"]),
        }

    def predict_xgboost(self, horizon: int) -> dict:
        """
        Run XGBoost prediction for a given horizon using the last available
        feature row from the feature matrix.

        Returns dict with flood_risk, drought_risk (int 0-3).
        """
        # Use last available row in feature matrix
        last_feat = self.fm.iloc[-1]
        X = last_feat[FEATURE_COLS].values.reshape(1, -1)

        flood_pred = 0
        drought_pred = 0

        if horizon in self.xgb_flood:
            flood_pred = int(self.xgb_flood[horizon].predict(X)[0])

        if horizon in self.xgb_drought:
            drought_pred = int(self.xgb_drought[horizon].predict(X)[0])

        return {
            "flood_risk": flood_pred,
            "drought_risk": drought_pred,
            "feature_date": self.fm.index[-1],
        }

    def predict_gencast_style(
        self, forecast_precip: np.ndarray, horizon: int
    ) -> dict:
        """
        Compute GenCast-style flood risk from forecast precipitation.

        Replicates the logic from postprocess_gencast_results.py:
          - API: forward-running from last known ERA5 API + forecast tp
          - SMI: held at last known ERA5 value (frozen)
          - total_ro: 0 (conservative assumption)
          - Flood score classified with training percentile thresholds

        Args:
            forecast_precip: Array of daily precipitation (m/day) for forecast days
            horizon: Target horizon (1, 3, or 7 days)

        Returns:
            dict with flood_risk per day and the target day's risk.
        """
        state = self.get_last_known_state()
        api_init = state["api_92"]
        smi_init = state["smi_fc"]
        spei_init = state["spei_6"]

        n_days = len(forecast_precip)

        # Forward-run API from last known state
        api_forecast = np.zeros(n_days)
        api_forecast[0] = API_DECAY * api_init + forecast_precip[0]
        for i in range(1, n_days):
            api_forecast[i] = API_DECAY * api_forecast[i - 1] + forecast_precip[i]

        # Compute flood score and risk for each forecast day
        flood_risks = []
        flood_scores = []
        drought_risks = []

        for d in range(n_days):
            score = self._calc_flood_score_scalar(api_forecast[d], smi_init, 0.0)
            flood_scores.append(score)
            flood_risks.append(self._flood_label(score))
            drought_risks.append(self._drought_label(spei_init, api_forecast[d], smi_init))

        # Target day (0-indexed: horizon-1)
        target_idx = min(horizon - 1, n_days - 1)

        return {
            "flood_risk": flood_risks[target_idx],
            "drought_risk": drought_risks[target_idx],
            "flood_risks_all": flood_risks,
            "flood_scores_all": flood_scores,
            "drought_risks_all": drought_risks,
            "api_forecast": api_forecast.tolist(),
        }

    def predict_hybrid(
        self, forecast_precip: np.ndarray, horizon: int
    ) -> dict:
        """
        Hybrid ensemble prediction: max(XGBoost, GenCast-style).

        This is the main prediction method that replicates the phase6
        hybrid ensemble logic: ensemble_risk = max(XGBoost_pred, GenCast_pred).

        Args:
            forecast_precip: Array of daily precipitation (m/day) for forecast days
            horizon: Target horizon (1, 3, or 7 days)

        Returns:
            dict with ensemble flood/drought risk, plus component predictions.
        """
        xgb_result = self.predict_xgboost(horizon)
        gm_result = self.predict_gencast_style(forecast_precip, horizon)

        # Hybrid ensemble: max-vote (from phase6 notebook)
        ensemble_flood = max(xgb_result["flood_risk"], gm_result["flood_risk"])

        # For drought: only XGBoost (GenCast cannot update SPEI-6)
        # But we include the threshold-based drought from GenCast-style as info
        ensemble_drought = xgb_result["drought_risk"]

        return {
            "ensemble_flood_risk": ensemble_flood,
            "ensemble_drought_risk": ensemble_drought,
            "xgb_flood_risk": xgb_result["flood_risk"],
            "xgb_drought_risk": xgb_result["drought_risk"],
            "gencast_flood_risk": gm_result["flood_risk"],
            "gencast_drought_risk": gm_result["drought_risk"],
            "flood_risks_all_days": gm_result["flood_risks_all"],
            "flood_scores_all_days": gm_result["flood_scores_all"],
            "drought_risks_all_days": gm_result["drought_risks_all"],
            "api_forecast": gm_result["api_forecast"],
            "horizon": horizon,
            "feature_date": xgb_result["feature_date"],
            "last_known_state": self.get_last_known_state(),
        }


def risk_description_flood(level: int, lang: str = "en") -> str:
    """Human-readable description of flood risk level."""
    descriptions = {
        "en": {
            0: "No elevated flood risk. Current indicators show normal conditions.",
            1: "Moderately elevated flood risk. Some chance of water nuisance, but the situation is not alarming.",
            2: "High flood risk. Indicators point to a significant chance of flooding.",
            3: "Extreme flood risk. Strong indication of possible flooding. Take precautionary measures immediately!",
        },
        "nl": {
            0: "Geen verhoogd overstromingsrisico. De huidige indicatoren wijzen op normale omstandigheden.",
            1: "Matig verhoogd overstromingsrisico. Er is enige kans op wateroverlast, maar de situatie is niet alarmerend.",
            2: "Hoog overstromingsrisico. De indicatoren wijzen op een aanzienlijke kans op wateroverlast of overstromingen.",
            3: "Extreem overstromingsrisico. Sterke indicatie van mogelijke overstromingen. Neem voorzorgsmaatregelen!",
        },
    }
    return descriptions.get(lang, descriptions["en"]).get(level, "Unknown risk level.")


def risk_description_drought(level: int, lang: str = "en") -> str:
    """Human-readable description of drought risk level."""
    descriptions = {
        "en": {
            0: "No elevated drought risk. Normal precipitation and soil moisture conditions.",
            1: "Moderately elevated drought risk. Mild signs of drought visible in indicators.",
            2: "High drought risk. Significant drought indicators. Water reserves may be under pressure.",
            3: "Extreme drought risk. Severe drought. Critical situation for water supply and agriculture.",
        },
        "nl": {
            0: "Geen verhoogd droogterisico. Normale neerslag- en bodemvochtomstandigheden.",
            1: "Matig verhoogd droogterisico. Lichte tekenen van droogte zichtbaar in de indicatoren.",
            2: "Hoog droogterisico. Significante droogte-indicatoren. Watervoorraden kunnen onder druk staan.",
            3: "Extreem droogterisico. Ernstige droogte. Kritieke situatie voor watervoorziening en landbouw.",
        },
    }
    return descriptions.get(lang, descriptions["en"]).get(level, "Unknown risk level.")
