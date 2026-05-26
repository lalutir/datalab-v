"""Time-series modeling module for ERA5 climate indices.

Provides OOP classes for preprocessing ERA5 daily data, fitting multiple
time-series forecasting models, and classifying drought and flood risk.

Forecasters
-----------
- SARIMAForecaster   : Seasonal ARIMA on monthly aggregated data.
- HoltWintersForecaster : Triple Exponential Smoothing on monthly data.
- RandomForestForecaster : Random Forest with engineered lag features.

Risk classifiers
----------------
- DroughtRiskClassifier : SPEI + API + SMI + Total Runoff -> 5-level drought risk.
- FloodRiskClassifier   : SPEI + API + SMI + Total Runoff -> 5-level flood risk.

References
----------
McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of
    drought frequency and duration to time scales. Preprints, 8th
    Conference on Applied Climatology, Anaheim, CA, Amer. Meteor. Soc., 179-184.

Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). A
    Multiscalar Drought Index Sensitive to Global Warming: The Standardized
    Precipitation Evapotranspiration Index. Journal of Climate, 23(7),
    1696–1718. https://doi.org/10.1175/2009JCLI2909.1

Kohler, M. A., & Linsley, R. K. (1951). Predicting the Runoff from Storm
    Rainfall. Research Note No. 34, U.S. Weather Bureau.

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
    Time Series Analysis: Forecasting and Control (5th ed.). Wiley.

Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and
    Practice (3rd ed.). OTexts. https://otexts.com/fpp3/

Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
    https://doi.org/10.1023/A:1010933404324
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Enumerations and data containers
# ---------------------------------------------------------------------------


class RiskLevel(Enum):
    """Five-tier risk classification for drought and flood events.

    Ordered from least to most severe. The ``numeric`` property maps each
    level to an integer in [0, 4] suitable for plotting or arithmetic.
    """

    LOW = "Laag risico"
    MODERATE = "Gemiddeld risico"
    ELEVATED = "Verhoogd risico"
    HIGH = "Hoog risico"
    EXTREME = "Extreem risico"

    @property
    def numeric(self) -> int:
        """Return integer representation (0 = LOW, 4 = EXTREME)."""
        return list(RiskLevel).index(self)

    @classmethod
    def all_labels(cls) -> List[str]:
        """Return all risk-level string values in order."""
        return [level.value for level in cls]


@dataclass
class ModelMetrics:
    """Container for forecast evaluation metrics.

    Attributes
    ----------
    model_name : str
        Name of the forecasting model.
    rmse : float
        Root Mean Squared Error.
    mae : float
        Mean Absolute Error.
    r2 : float
        Coefficient of determination (R²).
    """

    model_name: str
    rmse: float
    mae: float
    r2: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ModelMetrics({self.model_name}: "
            f"RMSE={self.rmse:.4f}, MAE={self.mae:.4f}, R²={self.r2:.4f})"
        )


@dataclass
class StationarityResult:
    """Container for an Augmented Dickey-Fuller stationarity test result.

    Attributes
    ----------
    series_name : str
        Name of the tested series.
    statistic : float
        ADF test statistic.
    p_value : float
        p-value of the test.
    is_stationary : bool
        True when p_value < significance level.
    critical_values : dict
        Critical values at 1 %, 5 %, and 10 % levels.
    """

    series_name: str
    statistic: float
    p_value: float
    is_stationary: bool
    critical_values: Dict[str, float]

    def __repr__(self) -> str:  # noqa: D105
        status = "stationary" if self.is_stationary else "non-stationary"
        return (
            f"StationarityResult({self.series_name}: {status}, "
            f"p={self.p_value:.4f})"
        )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TimeSeriesPreprocessor:
    """Load and prepare ERA5 Parquet data for time-series modelling.

    Handles loading, index conversion, optional temporal resampling,
    missing-value treatment, and train/test splitting.

    Parameters
    ----------
    data_path : str
        Path to the ERA5 Parquet file.
    resample_freq : str, optional
        Pandas offset alias for resampling (e.g. ``'ME'`` for month-end).
        When ``None`` the original daily frequency is retained.

    Examples
    --------
    >>> prep = TimeSeriesPreprocessor("era5_2000_2025.parquet", "ME")
    >>> prep.load()
    >>> train, test = prep.train_test_split("2023-01-01")
    """

    #: Columns that are target prediction variables.
    TARGET_COLUMNS: List[str] = ["spei", "api", "smi", "total_ro"]

    #: Aggregation rules applied during monthly resampling.
    _RESAMPLE_AGG: Dict[str, str] = {
        "t2m": "mean", "d2m": "mean", "u10": "mean", "v10": "mean",
        "sp": "mean", "swvl1": "mean", "swvl2": "mean", "ssrd": "sum",
        "tp": "sum", "e": "sum", "pev": "sum", "ssro": "sum",
        "sro": "sum", "lsp": "sum", "vimd": "sum", "total_ro": "sum",
        "api": "mean", "smi": "mean", "spei": "mean",
    }

    def __init__(
        self,
        data_path: str,
        resample_freq: Optional[str] = None,
    ) -> None:
        self.data_path = data_path
        self.resample_freq = resample_freq
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> "TimeSeriesPreprocessor":
        """Load the Parquet file and apply all preprocessing steps.

        Returns
        -------
        self
            Enables method chaining.
        """
        df = pd.read_parquet(self.data_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        df = df.drop(columns=["longitude", "latitude", "number"], errors="ignore")

        # SPEI has a 29-day warm-up period with NaN; back-fill from first valid value.
        df["spei"] = df["spei"].bfill()

        if self.resample_freq is not None:
            agg = {k: v for k, v in self._RESAMPLE_AGG.items() if k in df.columns}
            df = df.resample(self.resample_freq).agg(agg)

        self._df = df
        return self

    @property
    def df(self) -> pd.DataFrame:
        """Processed DataFrame (call :meth:`load` first)."""
        if self._df is None:
            raise RuntimeError("Call .load() before accessing .df.")
        return self._df

    def train_test_split(
        self, test_start: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into chronological train and test sets.

        Parameters
        ----------
        test_start : str
            ISO 8601 date string (e.g. ``'2023-01-01'``).

        Returns
        -------
        train : pd.DataFrame
        test : pd.DataFrame
        """
        cutoff = pd.Timestamp(test_start)
        return self._df[self._df.index < cutoff], self._df[self._df.index >= cutoff]


# ---------------------------------------------------------------------------
# Stationarity helpers
# ---------------------------------------------------------------------------


def check_stationarity(
    series: pd.Series,
    significance: float = 0.05,
) -> StationarityResult:
    """Run an Augmented Dickey-Fuller test on a time series.

    Parameters
    ----------
    series : pd.Series
        The time series to test (NaN values are dropped).
    significance : float
        Significance level for the stationarity decision (default ``0.05``).

    Returns
    -------
    StationarityResult
    """
    clean = series.dropna()
    adf_stat, p_value, _, _, critical_values, _ = adfuller(clean, autolag="AIC")
    return StationarityResult(
        series_name=series.name or "unknown",
        statistic=float(adf_stat),
        p_value=float(p_value),
        is_stationary=p_value < significance,
        critical_values={k: float(v) for k, v in critical_values.items()},
    )


def build_lag_features(
    series: pd.Series,
    lags: List[int],
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """Build a feature matrix of lagged values and optional seasonal encodings.

    Parameters
    ----------
    series : pd.Series
        Input time series with a DatetimeIndex.
    lags : list of int
        Lag periods to include as features.
    include_cyclical : bool
        When True (default), adds sine/cosine encodings of month and day-of-year.
        Set to False for a pure autoregressive feature set.

    Returns
    -------
    pd.DataFrame
        Feature matrix, dropping rows where any lag is NaN.
    """
    name = series.name or "value"
    df = series.to_frame(name=name)
    for lag in lags:
        df[f"lag_{lag}"] = df[name].shift(lag)
    if include_cyclical:
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    return df.dropna()


# ---------------------------------------------------------------------------
# Abstract base forecaster
# ---------------------------------------------------------------------------


class BaseForecaster(ABC):
    """Abstract base class for all time-series forecasters.

    Subclasses must implement :meth:`fit` and :meth:`predict`.
    Evaluation is handled by the shared :meth:`evaluate` method.

    Parameters
    ----------
    target_column : str
        Name of the target variable being modelled.
    """

    def __init__(self, target_column: str) -> None:
        self.target_column = target_column
        self._is_fitted: bool = False
        self._metrics: Optional[ModelMetrics] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, train: pd.Series) -> "BaseForecaster":
        """Fit the model on the training series.

        Parameters
        ----------
        train : pd.Series
            Training observations with a DatetimeIndex.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """Generate a multi-step-ahead out-of-sample forecast.

        Parameters
        ----------
        steps : int
            Number of future periods to forecast.

        Returns
        -------
        pd.Series
        """

    @abstractmethod
    def predict_in_sample(self, test: pd.Series) -> pd.Series:
        """Generate predictions aligned with an existing test series.

        Parameters
        ----------
        test : pd.Series
            Held-out observations used only for their index alignment.

        Returns
        -------
        pd.Series
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def evaluate(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> ModelMetrics:
        """Compute RMSE, MAE, and R² between truth and predictions.

        Parameters
        ----------
        y_true : pd.Series
            Ground-truth observations.
        y_pred : pd.Series
            Model predictions (must align with ``y_true``).

        Returns
        -------
        ModelMetrics
        """
        aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
        rmse = float(np.sqrt(mean_squared_error(aligned_true, aligned_pred)))
        mae = float(mean_absolute_error(aligned_true, aligned_pred))
        r2 = float(r2_score(aligned_true, aligned_pred))
        self._metrics = ModelMetrics(
            model_name=self.__class__.__name__,
            rmse=rmse,
            mae=mae,
            r2=r2,
        )
        return self._metrics

    @property
    def metrics(self) -> Optional[ModelMetrics]:
        """Last computed evaluation metrics, or ``None`` if not yet evaluated."""
        return self._metrics

    def _require_fitted(self) -> None:
        """Raise RuntimeError when the model has not been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling predict()."
            )


# ---------------------------------------------------------------------------
# Concrete forecasters
# ---------------------------------------------------------------------------


class SARIMAForecaster(BaseForecaster):
    """Seasonal ARIMA forecaster wrapping ``statsmodels`` SARIMAX.

    SARIMA is well-suited for monthly climate data with a strong annual
    cycle (seasonal period ``s=12``). The model decomposes the series into
    auto-regressive, integrated, and moving-average components at both the
    non-seasonal (p, d, q) and seasonal (P, D, Q) levels.

    Parameters
    ----------
    target_column : str
        Name of the target variable.
    order : tuple of int
        Non-seasonal ARIMA order ``(p, d, q)``.
    seasonal_order : tuple of int
        Seasonal ARIMA order ``(P, D, Q, s)``.

    References
    ----------
    Box, G. E. P. et al. (2015). Time Series Analysis (5th ed.). Wiley.
    """

    def __init__(
        self,
        target_column: str,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
    ) -> None:
        super().__init__(target_column)
        self.order = order
        self.seasonal_order = seasonal_order
        self._result = None

    def fit(self, train: pd.Series) -> "SARIMAForecaster":
        """Fit SARIMA on the monthly training series.

        Parameters
        ----------
        train : pd.Series
            Monthly training observations.

        Returns
        -------
        self
        """
        model = SARIMAX(
            train.dropna(),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)
        self._train = train
        self._is_fitted = True
        return self

    def predict(self, steps: int) -> pd.Series:
        """Forecast ``steps`` months ahead.

        Parameters
        ----------
        steps : int
            Number of future months.

        Returns
        -------
        pd.Series
        """
        self._require_fitted()
        return self._result.forecast(steps=steps)

    def predict_in_sample(self, test: pd.Series) -> pd.Series:
        """Forecast the test period (uses held-out length, not values).

        Parameters
        ----------
        test : pd.Series
            Test observations (only the index is used for alignment).

        Returns
        -------
        pd.Series
        """
        self._require_fitted()
        pred = self._result.forecast(steps=len(test))
        pred.index = test.index
        return pred


class HoltWintersForecaster(BaseForecaster):
    """Triple Exponential Smoothing (Holt-Winters) forecaster.

    Captures level, trend, and seasonal components without requiring
    stationarity. Works well for monthly data with a stable annual cycle.

    Parameters
    ----------
    target_column : str
        Name of the target variable.
    seasonal_periods : int
        Number of periods per season (12 for monthly data).
    trend : {'add', 'mul', None}
        Trend component type.
    seasonal : {'add', 'mul', None}
        Seasonal component type.

    References
    ----------
    Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles
        and Practice (3rd ed.). OTexts. https://otexts.com/fpp3/
    """

    def __init__(
        self,
        target_column: str,
        seasonal_periods: int = 12,
        trend: str = "add",
        seasonal: str = "add",
    ) -> None:
        super().__init__(target_column)
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self._result = None

    def fit(self, train: pd.Series) -> "HoltWintersForecaster":
        """Fit Holt-Winters on the training series.

        Parameters
        ----------
        train : pd.Series
            Training observations.

        Returns
        -------
        self
        """
        sp = self.seasonal_periods if self.seasonal is not None else None
        model = ExponentialSmoothing(
            train.dropna(),
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=sp,
            initialization_method="estimated",
        )
        self._result = model.fit(optimized=True)
        self._train = train
        self._is_fitted = True
        return self

    def predict(self, steps: int) -> pd.Series:
        """Forecast ``steps`` months ahead.

        Parameters
        ----------
        steps : int
            Number of future months.

        Returns
        -------
        pd.Series
        """
        self._require_fitted()
        return self._result.forecast(steps)

    def predict_in_sample(self, test: pd.Series) -> pd.Series:
        """Forecast the test period.

        Parameters
        ----------
        test : pd.Series
            Test observations (index used for alignment).

        Returns
        -------
        pd.Series
        """
        self._require_fitted()
        pred = self._result.forecast(len(test))
        pred.index = test.index
        return pred


class RandomForestForecaster(BaseForecaster):
    """Random Forest regressor for time-series via lag-feature engineering.

    Transforms the forecasting problem into supervised regression by using
    lagged values and cyclical seasonal encodings as features. Supports
    recursive multi-step-ahead prediction.

    Parameters
    ----------
    target_column : str
        Name of the target variable.
    lags : list of int, optional
        Lag periods used as input features.  Defaults to
        ``[1, 2, 3, 6, 12]`` (months) or ``[1, 7, 14, 30]`` (days).
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Seed for reproducibility.

    References
    ----------
    Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
        https://doi.org/10.1023/A:1010933404324
    """

    def __init__(
        self,
        target_column: str,
        lags: Optional[List[int]] = None,
        n_estimators: int = 200,
        random_state: int = 42,
        include_cyclical: bool = True,
    ) -> None:
        super().__init__(target_column)
        self.lags = lags or [1, 2, 3, 6, 12]
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.include_cyclical = include_cyclical
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._feature_cols: List[str] = []
        self._train: Optional[pd.Series] = None

    # ------------------------------------------------------------------

    def fit(self, train: pd.Series) -> "RandomForestForecaster":
        """Fit the Random Forest on engineered lag features.

        Parameters
        ----------
        train : pd.Series
            Training time series.

        Returns
        -------
        self
        """
        df = build_lag_features(train, self.lags, include_cyclical=self.include_cyclical)
        self._feature_cols = [c for c in df.columns if c != self.target_column]
        X = df[self._feature_cols].values
        y = df[self.target_column].values
        self._model.fit(X, y)
        self._train = train
        self._is_fitted = True
        return self

    def predict_in_sample(self, test: pd.Series) -> pd.Series:
        """Generate predictions for the test period using true lag values.

        The full combined series (train + test) is used to build lag
        features for the test window, so predictions use actual historical
        values rather than recursively accumulated errors.

        Parameters
        ----------
        test : pd.Series
            Test observations.

        Returns
        -------
        pd.Series
        """
        self._require_fitted()
        combined = pd.concat([self._train, test])
        df = build_lag_features(combined, self.lags, include_cyclical=self.include_cyclical)
        test_df = df.loc[test.index.intersection(df.index)]
        X = test_df[self._feature_cols].values
        preds = self._model.predict(X)
        return pd.Series(preds, index=test_df.index, name=self.target_column)

    def predict(self, steps: int) -> pd.Series:
        """Recursive multi-step-ahead forecast.

        Each predicted value is appended to the history and used as input
        for the next step.  Uncertainty accumulates with forecast horizon.

        Parameters
        ----------
        steps : int
            Number of periods to forecast ahead.

        Returns
        -------
        pd.Series
        """
        self._require_fitted()
        history = list(self._train.dropna().values)
        freq = pd.infer_freq(self._train.index) or "ME"
        last_date = self._train.index[-1]
        future_dates = pd.date_range(last_date, periods=steps + 1, freq=freq)[1:]
        max_lag = max(self.lags)

        preds = []
        for date in future_dates:
            lag_vals = [history[-(lag)] for lag in self.lags]
            if self.include_cyclical:
                month_sin = np.sin(2 * np.pi * date.month / 12)
                month_cos = np.cos(2 * np.pi * date.month / 12)
                doy_sin = np.sin(2 * np.pi * date.dayofyear / 365.25)
                doy_cos = np.cos(2 * np.pi * date.dayofyear / 365.25)
                features = lag_vals + [month_sin, month_cos, doy_sin, doy_cos]
            else:
                features = lag_vals
            X = np.array(features).reshape(1, -1)
            pred = float(self._model.predict(X)[0])
            preds.append(pred)
            history.append(pred)

        return pd.Series(preds, index=future_dates, name=self.target_column)

    @property
    def feature_importances(self) -> pd.Series:
        """Feature importances sorted descending (fitted model only)."""
        self._require_fitted()
        return (
            pd.Series(self._model.feature_importances_, index=self._feature_cols)
            .sort_values(ascending=False)
        )


class XGBoostForecaster(BaseForecaster):
    """XGBoost gradient-boosting forecaster via lag-feature engineering.

    Shares the same autoregressive feature set as ``RandomForestForecaster``
    but uses gradient-boosted trees, which often achieve lower bias through
    sequential error correction.

    Parameters
    ----------
    target_column : str
    lags : list of int
    n_estimators : int
    max_depth : int
    learning_rate : float
    random_state : int
    include_cyclical : bool

    References
    ----------
    Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
        System. KDD '16. https://doi.org/10.1145/2939672.2939785
    """

    def __init__(
        self,
        target_column: str,
        lags: Optional[List[int]] = None,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        include_cyclical: bool = True,
    ) -> None:
        super().__init__(target_column)
        self.lags = lags or [1, 2, 3, 6, 12]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.include_cyclical = include_cyclical
        self._feature_cols: List[str] = []
        self._train: Optional[pd.Series] = None
        self._model = None

    def _build_model(self):
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost is required: pip install xgboost")
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1,
        )

    def fit(self, train: pd.Series) -> "XGBoostForecaster":
        self._model = self._build_model()
        df = build_lag_features(train, self.lags, include_cyclical=self.include_cyclical)
        self._feature_cols = [c for c in df.columns if c != self.target_column]
        self._model.fit(df[self._feature_cols].values, df[self.target_column].values)
        self._train = train
        self._is_fitted = True
        return self

    def predict_in_sample(self, test: pd.Series) -> pd.Series:
        self._require_fitted()
        combined = pd.concat([self._train, test])
        df = build_lag_features(combined, self.lags, include_cyclical=self.include_cyclical)
        test_df = df.loc[test.index.intersection(df.index)]
        preds = self._model.predict(test_df[self._feature_cols].values)
        return pd.Series(preds, index=test_df.index, name=self.target_column)

    def predict(self, steps: int) -> pd.Series:
        self._require_fitted()
        history = list(self._train.dropna().values)
        freq = pd.infer_freq(self._train.index) or "D"
        last_date = self._train.index[-1]
        future_dates = pd.date_range(last_date, periods=steps + 1, freq=freq)[1:]

        preds = []
        for date in future_dates:
            lag_vals = [history[-(lag)] for lag in self.lags]
            if self.include_cyclical:
                features = lag_vals + [
                    np.sin(2 * np.pi * date.month / 12),
                    np.cos(2 * np.pi * date.month / 12),
                    np.sin(2 * np.pi * date.dayofyear / 365.25),
                    np.cos(2 * np.pi * date.dayofyear / 365.25),
                ]
            else:
                features = lag_vals
            pred = float(self._model.predict(np.array(features).reshape(1, -1))[0])
            preds.append(pred)
            history.append(pred)

        return pd.Series(preds, index=future_dates, name=self.target_column)

    @property
    def feature_importances(self) -> pd.Series:
        """Feature importances sorted descending."""
        self._require_fitted()
        return (
            pd.Series(self._model.feature_importances_, index=self._feature_cols)
            .sort_values(ascending=False)
        )


class LSTMForecaster(BaseForecaster):
    """LSTM (Long Short-Term Memory) forecaster using Keras/TensorFlow.

    Models the series as a sequence-to-one problem: a sliding window of
    ``lookback`` timesteps is used to predict the next value. Multi-step
    predictions are generated recursively. Values are Min-Max scaled before
    training and inverse-transformed on output.

    Parameters
    ----------
    target_column : str
    lookback : int
        Number of past timesteps used as input sequence (default 30).
    units : int
        Number of LSTM units in the hidden layer.
    epochs : int
    batch_size : int
    random_state : int

    References
    ----------
    Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
        Neural Computation, 9(8), 1735–1780.
    """

    def __init__(
        self,
        target_column: str,
        lookback: int = 30,
        units: int = 64,
        epochs: int = 20,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        super().__init__(target_column)
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._scaler = MinMaxScaler()
        self._keras_model = None
        self._train: Optional[pd.Series] = None

    @staticmethod
    def _build_sequences(values: np.ndarray, lookback: int):
        X, y = [], []
        for i in range(lookback, len(values)):
            X.append(values[i - lookback:i])
            y.append(values[i])
        return np.array(X)[..., np.newaxis], np.array(y)

    def fit(self, train: pd.Series) -> "LSTMForecaster":
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except ImportError:
            raise ImportError("tensorflow is required: pip install tensorflow")

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        values = train.dropna().values.reshape(-1, 1)
        scaled = self._scaler.fit_transform(values).flatten()
        X, y = self._build_sequences(scaled, self.lookback)

        model = Sequential([
            LSTM(self.units, input_shape=(self.lookback, 1)),
            Dropout(0.1),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        self._keras_model = model
        self._train = train
        self._is_fitted = True
        return self

    def predict_in_sample(self, test: pd.Series) -> pd.Series:
        self._require_fitted()
        combined = pd.concat([self._train, test]).dropna()
        scaled = self._scaler.transform(combined.values.reshape(-1, 1)).flatten()
        train_len = len(self._train.dropna())

        preds_scaled = []
        for i in range(train_len, len(scaled)):
            seq = scaled[i - self.lookback:i].reshape(1, self.lookback, 1)
            preds_scaled.append(float(self._keras_model.predict(seq, verbose=0)[0][0]))

        preds = self._scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        return pd.Series(preds, index=test.index[:len(preds)], name=self.target_column)

    def predict(self, steps: int) -> pd.Series:
        self._require_fitted()
        scaled = self._scaler.transform(
            self._train.dropna().values.reshape(-1, 1)
        ).flatten().tolist()

        freq = pd.infer_freq(self._train.index) or "D"
        last_date = self._train.index[-1]
        future_dates = pd.date_range(last_date, periods=steps + 1, freq=freq)[1:]

        preds_scaled = []
        for _ in future_dates:
            seq = np.array(scaled[-self.lookback:]).reshape(1, self.lookback, 1)
            pred = float(self._keras_model.predict(seq, verbose=0)[0][0])
            preds_scaled.append(pred)
            scaled.append(pred)

        preds = self._scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        return pd.Series(preds, index=future_dates, name=self.target_column)


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet forecaster for trend + seasonality decomposition.

    Prophet decomposes the series into trend, yearly seasonality, and optional
    weekly seasonality. Well-suited for daily climate indices with strong annual
    cycles. Predictions are made directly for requested future dates without
    recursive accumulation of error.

    Parameters
    ----------
    target_column : str
    yearly_seasonality : bool
    weekly_seasonality : bool
    changepoint_prior_scale : float
        Controls trend flexibility; higher values allow more breakpoints.

    References
    ----------
    Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. The American
        Statistician, 72(1), 37–45. https://doi.org/10.1080/00031305.2017.1380080
    """

    def __init__(
        self,
        target_column: str,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
    ) -> None:
        super().__init__(target_column)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self._prophet_model = None
        self._train: Optional[pd.Series] = None

    def fit(self, train: pd.Series) -> "ProphetForecaster":
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet is required: pip install prophet")

        df = train.dropna().reset_index()
        df.columns = ["ds", "y"]

        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        model.fit(df)

        self._prophet_model = model
        self._train = train
        self._is_fitted = True
        return self

    def predict_in_sample(self, test: pd.Series) -> pd.Series:
        self._require_fitted()
        future = pd.DataFrame({"ds": test.index})
        forecast = self._prophet_model.predict(future)
        return pd.Series(forecast["yhat"].values, index=test.index, name=self.target_column)

    def predict(self, steps: int) -> pd.Series:
        self._require_fitted()
        freq = pd.infer_freq(self._train.index) or "D"
        last_date = self._train.index[-1]
        future_dates = pd.date_range(last_date, periods=steps + 1, freq=freq)[1:]
        forecast = self._prophet_model.predict(pd.DataFrame({"ds": future_dates}))
        return pd.Series(forecast["yhat"].values, index=future_dates, name=self.target_column)


# ---------------------------------------------------------------------------
# Risk classifiers
# ---------------------------------------------------------------------------


class DroughtRiskClassifier:
    """Classify drought risk from SPEI, API, SMI, and Total Runoff.

    SPEI is the primary indicator, classified according to the McKee et al.
    (1993) scale.  API, SMI, and Total Runoff act as secondary modifiers:
    when any of them falls below its historical low-percentile threshold,
    risk is elevated one level because persistently low antecedent moisture
    and soil water amplify drought severity.

    Risk mapping (SPEI primary):

    +--------------------+-------------------+
    | SPEI range         | Base risk level   |
    +====================+===================+
    | >= -0.5            | Laag              |
    | -1.0 to -0.5       | Gemiddeld         |
    | -1.5 to -1.0       | Verhoogd          |
    | -2.0 to -1.5       | Hoog              |
    | < -2.0             | Extreem           |
    +--------------------+-------------------+

    Parameters
    ----------
    api_low_percentile : float
        Quantile (0–1) of API below which drought risk is elevated.
    smi_low_percentile : float
        Quantile (0–1) of SMI below which drought risk is elevated.
    ro_low_percentile : float
        Quantile (0–1) of total_ro below which drought risk is elevated.

    References
    ----------
    McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of
        drought frequency and duration to time scales.

    Vicente-Serrano, S. M. et al. (2010). A Multiscalar Drought Index.
        Journal of Climate, 23(7), 1696–1718.

    Kohler, M. A., & Linsley, R. K. (1951). Predicting the Runoff from
        Storm Rainfall. U.S. Weather Bureau Research Note No. 34.
    """

    _SPEI_THRESHOLDS: List[Tuple[float, RiskLevel]] = [
        (-2.0, RiskLevel.EXTREME),
        (-1.5, RiskLevel.HIGH),
        (-1.0, RiskLevel.ELEVATED),
        (-0.5, RiskLevel.MODERATE),
    ]

    def __init__(
        self,
        api_low_percentile: float = 0.25,
        smi_low_percentile: float = 0.25,
        ro_low_percentile: float = 0.25,
    ) -> None:
        self.api_low_percentile = api_low_percentile
        self.smi_low_percentile = smi_low_percentile
        self.ro_low_percentile = ro_low_percentile
        self._api_cutoff: Optional[float] = None
        self._smi_cutoff: Optional[float] = None
        self._ro_cutoff: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "DroughtRiskClassifier":
        """Compute secondary-indicator cutoffs from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame with ``'api'``, ``'smi'``, and
            ``'total_ro'`` columns.

        Returns
        -------
        self
        """
        self._api_cutoff = float(df["api"].quantile(self.api_low_percentile))
        self._smi_cutoff = float(df["smi"].quantile(self.smi_low_percentile))
        self._ro_cutoff = float(df["total_ro"].quantile(self.ro_low_percentile))
        return self

    @staticmethod
    def _spei_to_base_risk(spei: float) -> RiskLevel:
        """Map a scalar SPEI value to a base RiskLevel."""
        for threshold, level in DroughtRiskClassifier._SPEI_THRESHOLDS:
            if spei < threshold:
                return level
        return RiskLevel.LOW

    @staticmethod
    def _elevate(risk: RiskLevel) -> RiskLevel:
        """Increase risk by one level (capped at EXTREME)."""
        levels = list(RiskLevel)
        idx = min(levels.index(risk) + 1, len(levels) - 1)
        return levels[idx]

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Classify drought risk row-by-row.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with ``'spei'``, ``'api'``, ``'smi'``, and
            ``'total_ro'`` columns.

        Returns
        -------
        pd.Series of RiskLevel
            Index matches ``df.index``.
        """
        if self._api_cutoff is None:
            raise RuntimeError("Call .fit() before .classify().")

        risks = []
        for _, row in df[["spei", "api", "smi", "total_ro"]].iterrows():
            risk = self._spei_to_base_risk(row["spei"])
            # Elevate if any secondary indicator signals dry conditions.
            # Skip when SPEI is already LOW or EXTREME to avoid inflation.
            if risk not in (RiskLevel.LOW, RiskLevel.EXTREME) and any([
                row["api"] < self._api_cutoff,
                row["smi"] < self._smi_cutoff,
                row["total_ro"] < self._ro_cutoff,
            ]):
                risk = self._elevate(risk)
            risks.append(risk)
        return pd.Series(risks, index=df.index, name="drought_risk")

    def classify_from_predictions(
        self,
        spei_pred: pd.Series,
        api_pred: pd.Series,
        smi_pred: pd.Series,
        total_ro_pred: pd.Series,
    ) -> pd.Series:
        """Classify drought risk from predicted index series.

        Parameters
        ----------
        spei_pred : pd.Series
            Predicted SPEI values.
        api_pred : pd.Series
            Predicted API values.
        smi_pred : pd.Series
            Predicted SMI values.
        total_ro_pred : pd.Series
            Predicted Total Runoff values.

        Returns
        -------
        pd.Series of RiskLevel
        """
        df = pd.DataFrame({
            "spei": spei_pred,
            "api": api_pred,
            "smi": smi_pred,
            "total_ro": total_ro_pred,
        })
        return self.classify(df)


class FloodRiskClassifier:
    """Classify flood risk from SPEI, API, SMI, and Total Runoff.

    A composite flood score is computed as a weighted average of the
    Min-Max normalised values of all four indices.  Score thresholds are
    derived from the empirical percentile distribution of the training data,
    making the classification adaptive to local hydro-climatic conditions.

    Composite score:
        flood_score = (spei_weight * norm_spei + api_weight * norm_api
                       + smi_weight * norm_smi + runoff_weight * norm_total_ro)

    Parameters
    ----------
    spei_weight : float
        Weight assigned to SPEI in the composite score (default: 0.25).
    api_weight : float
        Weight assigned to API in the composite score (default: 0.25).
    smi_weight : float
        Weight assigned to SMI in the composite score (default: 0.25).
    runoff_weight : float
        Weight assigned to total runoff in the composite score (default: 0.25).
    percentile_thresholds : dict, optional
        Mapping of ``{RiskLevel: quantile}`` that defines score breakpoints.

    References
    ----------
    Berghuijs, W. R., et al. (2019). Dominant flood generating mechanisms
        across the United States. Geophysical Research Letters, 43(9),
        4382–4390. https://doi.org/10.1002/2016GL068070

    Martens, B. et al. (2017). GLEAM v3: satellite-based land evaporation
        and root-zone soil moisture. Geoscientific Model Development, 10,
        1903–1925. https://doi.org/10.5194/gmd-10-1903-2017
    """

    _DEFAULT_PERCENTILES: Dict[RiskLevel, float] = {
        RiskLevel.LOW: 0.65,
        RiskLevel.MODERATE: 0.80,
        RiskLevel.ELEVATED: 0.90,
        RiskLevel.HIGH: 0.97,
        RiskLevel.EXTREME: 1.00,
    }

    def __init__(
        self,
        spei_weight: float = 0.25,
        api_weight: float = 0.25,
        smi_weight: float = 0.25,
        runoff_weight: float = 0.25,
        percentile_thresholds: Optional[Dict[RiskLevel, float]] = None,
    ) -> None:
        self.spei_weight = spei_weight
        self.api_weight = api_weight
        self.smi_weight = smi_weight
        self.runoff_weight = runoff_weight
        self._percentile_thresholds = (
            percentile_thresholds or self._DEFAULT_PERCENTILES
        )
        self._spei_scaler = MinMaxScaler()
        self._api_scaler = MinMaxScaler()
        self._smi_scaler = MinMaxScaler()
        self._ro_scaler = MinMaxScaler()
        self._score_cutoffs: Optional[Dict[RiskLevel, float]] = None

    def fit(self, df: pd.DataFrame) -> "FloodRiskClassifier":
        """Fit normalisation scalers and score thresholds on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame with ``'spei'``, ``'api'``, ``'smi'``, and
            ``'total_ro'`` columns.

        Returns
        -------
        self
        """
        self._spei_scaler.fit(df[["spei"]])
        self._api_scaler.fit(df[["api"]])
        self._smi_scaler.fit(df[["smi"]])
        self._ro_scaler.fit(df[["total_ro"]])
        scores = self._compute_scores(df)
        self._score_cutoffs = {
            risk: float(scores.quantile(q))
            for risk, q in self._percentile_thresholds.items()
        }
        return self

    def _compute_scores(self, df: pd.DataFrame) -> pd.Series:
        """Compute the composite flood score for each row."""
        norm_spei = self._spei_scaler.transform(df[["spei"]]).flatten()
        norm_api = self._api_scaler.transform(df[["api"]]).flatten()
        norm_smi = self._smi_scaler.transform(df[["smi"]]).flatten()
        norm_ro = self._ro_scaler.transform(df[["total_ro"]]).flatten()
        return pd.Series(
            self.spei_weight * norm_spei
            + self.api_weight * norm_api
            + self.smi_weight * norm_smi
            + self.runoff_weight * norm_ro,
            index=df.index,
        )

    def _score_to_risk(self, score: float) -> RiskLevel:
        """Map a composite score to a RiskLevel using fitted thresholds."""
        for risk, cutoff in sorted(
            self._score_cutoffs.items(), key=lambda x: x[1]
        ):
            if score <= cutoff:
                return risk
        return RiskLevel.EXTREME

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Classify flood risk row-by-row.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with ``'spei'``, ``'api'``, ``'smi'``, and
            ``'total_ro'`` columns.

        Returns
        -------
        pd.Series of RiskLevel
        """
        if self._score_cutoffs is None:
            raise RuntimeError("Call .fit() before .classify().")
        scores = self._compute_scores(df)
        return scores.apply(self._score_to_risk).rename("flood_risk")

    def classify_from_predictions(
        self,
        spei_pred: pd.Series,
        api_pred: pd.Series,
        smi_pred: pd.Series,
        total_ro_pred: pd.Series,
    ) -> pd.Series:
        """Classify flood risk from predicted index series.

        Parameters
        ----------
        spei_pred : pd.Series
            Predicted SPEI values.
        api_pred : pd.Series
            Predicted API values.
        smi_pred : pd.Series
            Predicted SMI values.
        total_ro_pred : pd.Series
            Predicted Total Runoff values.

        Returns
        -------
        pd.Series of RiskLevel
        """
        df = pd.DataFrame({
            "spei": spei_pred,
            "api": api_pred,
            "smi": smi_pred,
            "total_ro": total_ro_pred,
        })
        return self.classify(df)
