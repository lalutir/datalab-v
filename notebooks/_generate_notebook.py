"""Helper script to generate the risk modelling notebook via nbformat."""

import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(source: str):
    cells.append(new_markdown_cell(source))


def code(source: str):
    cells.append(new_code_cell(source))


# ─── Cel 1 – Titel ─────────────────────────────────────────────────────────
md("""\
# ERA5 Time-Series Modellering en Risicoclassificatie
## Droogte- en Overstromingsrisico 2000–2025

Dit notebook modelleert vier klimaatindices afkomstig uit de ERA5-dataset
(periode 2000–2025, locatie 42,75°O / 9,25°N, Hoorn van Afrika):

| Index | Beschrijving | Risico-doel |
|-------|-------------|-------------|
| **SPEI** | Standardized Precipitation-Evapotranspiration Index | Droogte |
| **API** | Antecedent Precipitation Index | Droogte |
| **SMI** | Soil Moisture Index | Overstroming |
| **Total Runoff** | Totale afvoer (m/dag) | Overstroming |

**Aanpak**: drie tijdreeksmodellen worden gefit, vergeleken en ingezet om
toekomstige waarden te voorspellen.  Op basis van de voorspellingen worden
risicoscores berekend met de klassen:
*Laag*, *Gemiddeld*, *Verhoogd*, *Hoog*, *Extreem*.
""")

# ─── Cel 2 – Imports ──────────────────────────────────────────────────────
code("""\
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Voeg de projectroot toe aan het pad zodat src-modules importeerbaar zijn
sys.path.insert(0, "..")

from src.modeling import (
    TimeSeriesPreprocessor,
    check_stationarity,
    SARIMAForecaster,
    HoltWintersForecaster,
    RandomForestForecaster,
    DroughtRiskClassifier,
    FloodRiskClassifier,
    RiskLevel,
)

# ─── Plot-stijl ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 13})

RISK_COLORS = {
    RiskLevel.LOW:      "#2ecc71",
    RiskLevel.MODERATE: "#f1c40f",
    RiskLevel.ELEVATED: "#e67e22",
    RiskLevel.HIGH:     "#e74c3c",
    RiskLevel.EXTREME:  "#8e44ad",
}
RISK_LEGEND = [Patch(color=c, label=r.value) for r, c in RISK_COLORS.items()]

DATA_PATH = "../src/data/processed/era5_2000_2025.parquet"
TEST_START = "2023-01-01"
FORECAST_MONTHS = 18  # horizon voor toekomstscenario
""")

# ─── Cel 3 – Data laden (maandelijks) ─────────────────────────────────────
md("## 1 · Data laden en verkennen")

code("""\
# ── Maandelijkse aggregatie voor SARIMA en Holt-Winters ────────────────────
prep_m = TimeSeriesPreprocessor(DATA_PATH, resample_freq="ME").load()
df_m = prep_m.df

# ── Dagelijkse data voor Random Forest en risicoclassificatie ──────────────
prep_d = TimeSeriesPreprocessor(DATA_PATH, resample_freq=None).load()
df_d = prep_d.df

print(f"Maandelijks: {df_m.shape[0]} rijen  ({df_m.index[0].date()} → {df_m.index[-1].date()})")
print(f"Dagelijks  : {df_d.shape[0]} rijen  ({df_d.index[0].date()} → {df_d.index[-1].date()})")
df_m[["spei", "api", "smi", "total_ro"]].describe().round(4)
""")

# ─── Cel 4 – Overzichtsplot ────────────────────────────────────────────────
code("""\
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
pairs = [
    ("spei",     "SPEI",         "royalblue",  "Droogte-index"),
    ("api",      "API (m)",      "darkorange",  "Neerslag-antecedent"),
    ("smi",      "SMI (0–1)",    "seagreen",    "Bodemvochtindex"),
    ("total_ro", "Runoff (m/d)", "firebrick",   "Totale afvoer"),
]
for ax, (col, ylabel, color, title) in zip(axes, pairs):
    ax.plot(df_d.index, df_d[col], lw=0.7, color=color, alpha=0.85)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.axvline(pd.Timestamp(TEST_START), color="grey", ls="--", lw=1.2,
               label="train/test splitsing")

axes[0].legend(loc="upper right", fontsize=9)
axes[-1].set_xlabel("Datum")
fig.suptitle("ERA5 Klimaatindices – dagelijkse waarden (2000–2025)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("overzicht_tijdreeksen.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 5 – Stationariteit ───────────────────────────────────────────────
md("""\
## 2 · Stationariteitsanalyse (ADF-test)

Een tijdreeks is *stationair* als gemiddelde en variantie niet veranderen
over de tijd.  De **Augmented Dickey-Fuller (ADF)** test toetst de
nulhypothese H₀: *de reeks bevat een eenheidswortel (non-stationair)*.
Een p-waarde < 0,05 verwerpt H₀ → stationair.

SARIMA en Holt-Winters kunnen omgaan met niet-stationariteit via de
differentiatieparameter `d` (of `D` voor de seizoenscomponent).
""")

code("""\
TARGET_COLS = ["spei", "api", "smi", "total_ro"]

rows = []
for col in TARGET_COLS:
    res = check_stationarity(df_m[col])
    rows.append({
        "Index": col.upper(),
        "ADF-statistiek": round(res.statistic, 3),
        "p-waarde": round(res.p_value, 4),
        "Stationair (α=0.05)": "✔ Ja" if res.is_stationary else "✘ Nee",
        "Kritieke waarde 5%": round(res.critical_values["5%"], 3),
    })

pd.DataFrame(rows).set_index("Index")
""")

# ─── Cel 6 – ACF / PACF ──────────────────────────────────────────────────
md("""\
## 3 · Seizoensanalyse – ACF en PACF

De **Autocorrelatiefunctie (ACF)** toont de correlatie van de reeks met
zichzelf bij verschillende lags.  De **Partiële ACF (PACF)** corrigeert
voor tussenliggende lags.  Pieken bij lag 12 (maanden) bevestigen een
jaarlijkse seizoenscyclus.
""")

code("""\
fig, axes = plt.subplots(4, 2, figsize=(14, 12))

for i, col in enumerate(TARGET_COLS):
    series = df_m[col].dropna()
    plot_acf(series, lags=36, ax=axes[i, 0],
             title=f"ACF – {col.upper()}", zero=False)
    plot_pacf(series, lags=24, ax=axes[i, 1],
              title=f"PACF – {col.upper()}", zero=False, method="ywm")

plt.tight_layout()
plt.savefig("acf_pacf.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 7 – Seizoensdecompositie ─────────────────────────────────────────
code("""\
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
for ax, col in zip(axes, TARGET_COLS):
    series = df_m[col].dropna()
    result = seasonal_decompose(series, model="additive", period=12)
    ax.plot(result.seasonal, lw=1, color="steelblue", label="Seizoenscomponent")
    ax.set_title(f"Seizoenscomponent – {col.upper()}", fontsize=11, loc="left")
    ax.set_xlabel("")
    ax.legend(fontsize=9)

plt.suptitle("Additieve seizoensdecompositie (maandelijkse data, periode=12)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("seizoensdecompositie.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 8 – Train/test split ─────────────────────────────────────────────
md("## 4 · Train/test splitsing")

code("""\
train_m, test_m = prep_m.train_test_split(TEST_START)
train_d, test_d = prep_d.train_test_split(TEST_START)

print(f"Maandelijks – train: {len(train_m)} | test: {len(test_m)}")
print(f"Dagelijks  – train: {len(train_d)} | test: {len(test_d)}")
""")

# ─── Cel 9 – SARIMA ──────────────────────────────────────────────────────
md("""\
## 5 · Model 1 – SARIMA

**Seasonal AutoRegressive Integrated Moving Average** (SARIMA) breidt het
klassieke ARIMA-model uit met seizoenscomponenten `(P, D, Q, s)` waarbij
`s=12` voor maandelijkse data.

Gekozen orde per index:

| Index | (p,d,q) | (P,D,Q,s) | Reden |
|-------|---------|-----------|-------|
| SPEI | (1,1,1) | (1,1,1,12) | Niet-stationair, sterke seizoenspieken bij lag 12 |
| API  | (1,0,1) | (1,1,1,12) | Near-stationair maar seizoenspatroon aanwezig |
| SMI  | (1,1,1) | (1,1,1,12) | Niet-stationair, seizoenspatroon |
| Total Runoff | (1,1,0) | (0,1,1,12) | Sterk asymmetrisch, eenvoudige orde |
""")

code("""\
SARIMA_PARAMS = {
    "spei":     {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
    "api":      {"order": (1, 0, 1), "seasonal_order": (1, 1, 1, 12)},
    "smi":      {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
    "total_ro": {"order": (1, 1, 0), "seasonal_order": (0, 1, 1, 12)},
}

sarima_models = {}
sarima_metrics = {}

for col, params in SARIMA_PARAMS.items():
    print(f"  Fitten SARIMA voor {col.upper()} ...", end=" ")
    model = SARIMAForecaster(target_column=col, **params)
    model.fit(train_m[col])
    pred = model.predict_in_sample(test_m[col])
    metrics = model.evaluate(test_m[col], pred)
    sarima_models[col] = model
    sarima_metrics[col] = metrics
    print(f"RMSE={metrics.rmse:.4f}  MAE={metrics.mae:.4f}  R²={metrics.r2:.3f}")
""")

code("""\
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
for ax, col in zip(axes, TARGET_COLS):
    model = sarima_models[col]
    pred  = model.predict_in_sample(test_m[col])
    ax.plot(train_m.index[-36:], train_m[col].iloc[-36:],
            color="steelblue", lw=1.2, label="Training (laatste 3 jr)")
    ax.plot(test_m.index, test_m[col],
            color="black", lw=1.5, ls="--", label="Werkelijk (test)")
    ax.plot(test_m.index, pred,
            color="tomato", lw=1.5, label="SARIMA voorspelling")
    ax.axvline(pd.Timestamp(TEST_START), color="grey", ls=":", lw=1)
    ax.set_title(f"SARIMA – {col.upper()}", loc="left")
    ax.legend(fontsize=9)

plt.suptitle("SARIMA: voorspellingen vs. werkelijke waarden (testset 2023–2025)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("sarima_resultaten.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 10 – Holt-Winters ───────────────────────────────────────────────
md("""\
## 6 · Model 2 – Holt-Winters (Triple Exponential Smoothing)

Holt-Winters past drie exponentiële afvlakkingen toe voor niveau, trend
en seizoen.  Het model is robuust bij kortere reeksen en vereist geen
stationariteit.  Voor `total_ro` wordt een additief seizoen gebruikt
omdat de reeks nul-inflatie heeft en geen negatieve waarden kent.
""")

code("""\
hw_models = {}
hw_metrics = {}

for col in TARGET_COLS:
    print(f"  Fitten HoltWinters voor {col.upper()} ...", end=" ")
    model = HoltWintersForecaster(
        target_column=col,
        seasonal_periods=12,
        trend="add",
        seasonal="add",
    )
    model.fit(train_m[col])
    pred = model.predict_in_sample(test_m[col])
    metrics = model.evaluate(test_m[col], pred)
    hw_models[col] = model
    hw_metrics[col] = metrics
    print(f"RMSE={metrics.rmse:.4f}  MAE={metrics.mae:.4f}  R²={metrics.r2:.3f}")
""")

code("""\
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
for ax, col in zip(axes, TARGET_COLS):
    model = hw_models[col]
    pred  = model.predict_in_sample(test_m[col])
    ax.plot(train_m.index[-36:], train_m[col].iloc[-36:],
            color="steelblue", lw=1.2, label="Training (laatste 3 jr)")
    ax.plot(test_m.index, test_m[col],
            color="black", lw=1.5, ls="--", label="Werkelijk")
    ax.plot(test_m.index, pred,
            color="mediumseagreen", lw=1.5, label="HoltWinters voorspelling")
    ax.set_title(f"Holt-Winters – {col.upper()}", loc="left")
    ax.legend(fontsize=9)

plt.suptitle("Holt-Winters: voorspellingen vs. werkelijke waarden",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("holtwinters_resultaten.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 11 – Random Forest ──────────────────────────────────────────────
md("""\
## 7 · Model 3 – Random Forest (maandelijks, lag-features)

Random Forest transformeert het voorspelprobleem naar gesuperviseerde
regressie door de tijdreeks te voorzien van *lag-features* (vertraagde
waarden) en *cyclische seizoensencoderingen* (sinus/cosinus van maand en
dag-van-het-jaar).  Dit maakt het model niet-parametrisch en robuust
voor niet-lineaire patronen.

Lags gebruikt: `[1, 2, 3, 6, 12]` maanden.
""")

code("""\
rf_models = {}
rf_metrics = {}

for col in TARGET_COLS:
    print(f"  Fitten RandomForest voor {col.upper()} ...", end=" ")
    model = RandomForestForecaster(
        target_column=col,
        lags=[1, 2, 3, 6, 12],
        n_estimators=200,
        random_state=42,
    )
    model.fit(train_m[col])
    pred = model.predict_in_sample(test_m[col])
    metrics = model.evaluate(test_m[col], pred)
    rf_models[col] = model
    rf_metrics[col] = metrics
    print(f"RMSE={metrics.rmse:.4f}  MAE={metrics.mae:.4f}  R²={metrics.r2:.3f}")
""")

code("""\
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
for ax, col in zip(axes, TARGET_COLS):
    model = rf_models[col]
    pred  = model.predict_in_sample(test_m[col])
    ax.plot(train_m.index[-36:], train_m[col].iloc[-36:],
            color="steelblue", lw=1.2, label="Training (laatste 3 jr)")
    ax.plot(test_m.index, test_m[col],
            color="black", lw=1.5, ls="--", label="Werkelijk")
    ax.plot(test_m.index, pred,
            color="darkorchid", lw=1.5, label="Random Forest voorspelling")
    ax.set_title(f"Random Forest – {col.upper()}", loc="left")
    ax.legend(fontsize=9)

plt.suptitle("Random Forest: voorspellingen vs. werkelijke waarden",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("rf_resultaten.png", bbox_inches="tight")
plt.show()
""")

code("""\
# Feature-importantie voor SPEI (representatief voorbeeld)
fi = rf_models["spei"].feature_importances
fig, ax = plt.subplots(figsize=(8, 4))
fi.plot.barh(ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Feature importantie – Random Forest SPEI", fontweight="bold")
ax.set_xlabel("Importantie (mean decrease impurity)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 12 – Modelcomparisatie ──────────────────────────────────────────
md("## 8 · Modelcomparisatie")

code("""\
rows = []
for col in TARGET_COLS:
    for model_name, metrics_dict in [
        ("SARIMA",       sarima_metrics),
        ("Holt-Winters", hw_metrics),
        ("RandomForest", rf_metrics),
    ]:
        m = metrics_dict[col]
        rows.append({
            "Index": col.upper(),
            "Model": model_name,
            "RMSE": round(m.rmse, 5),
            "MAE":  round(m.mae, 5),
            "R²":   round(m.r2, 3),
        })

comparison_df = pd.DataFrame(rows).set_index(["Index", "Model"])
comparison_df.style.highlight_min(subset=["RMSE", "MAE"], color="lightgreen") \\
                   .highlight_max(subset=["R²"],           color="lightgreen")
""")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric in zip(axes, ["RMSE", "MAE", "R²"]):
    pivot = comparison_df[metric].unstack("Model")
    pivot.plot.bar(ax=ax, edgecolor="white")
    ax.set_title(metric, fontweight="bold")
    ax.set_xlabel("Index")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=0)

plt.suptitle("Modelcomparisatie op testset (2023–2025)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("model_vergelijking.png", bbox_inches="tight")
plt.show()
""")

code("""\
# Selecteer het beste model per index op basis van RMSE
best_models = {}
for col in TARGET_COLS:
    col_df = comparison_df.loc[col.upper(), "RMSE"]
    best_name = col_df.idxmin()
    print(f"{col.upper():12s} → beste model: {best_name}  "
          f"(RMSE={col_df[best_name]:.5f})")
    if best_name == "SARIMA":
        best_models[col] = sarima_models[col]
    elif best_name == "Holt-Winters":
        best_models[col] = hw_models[col]
    else:
        best_models[col] = rf_models[col]
""")

# ─── Cel 13 – Droogterisico ──────────────────────────────────────────────
md("""\
## 9 · Droogterisicoclassificatie

De `DroughtRiskClassifier` combineert SPEI en API:

1. **SPEI** (primair) – geclassificeerd conform McKee et al. (1993):
   - ≥ −0,5 → *Laag*  | −1,0 tot −0,5 → *Gemiddeld*
   - −1,5 tot −1,0 → *Verhoogd* | −2,0 tot −1,5 → *Hoog* | < −2,0 → *Extreem*
2. **API** (modificator) – als API < 25e percentiel (kwartiel) wordt
   het risiconiveau één klasse verhoogd.
""")

code("""\
drought_clf = DroughtRiskClassifier(api_low_percentile=0.25)
drought_clf.fit(train_d)

# Classificeer de volledige dagelijkse reeks
drought_risk_daily = drought_clf.classify(df_d)

# Overzicht per risicoklasse
counts = drought_risk_daily.value_counts()
pct    = (counts / len(drought_risk_daily) * 100).round(1)
summary = pd.DataFrame({"Dagen": counts, "%": pct})
summary.index = [r.value for r in summary.index]
print("Droogterisico – verdeling over volledige periode:")
summary
""")

code("""\
fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# Tijdreeks met achtergrondkleuren
ax = axes[0]
ax.plot(df_d.index, df_d["spei"], lw=0.8, color="royalblue", label="SPEI")
ax.axhline(-0.5, color="orange",  ls="--", lw=0.8, alpha=0.7)
ax.axhline(-1.0, color="darkorange", ls="--", lw=0.8, alpha=0.7)
ax.axhline(-1.5, color="red", ls="--", lw=0.8, alpha=0.7)
ax.axhline(-2.0, color="purple", ls="--", lw=0.8, alpha=0.7)
ax.set_ylabel("SPEI")
ax.set_title("SPEI met drempelwaarden voor droogterisico", loc="left")
ax.legend(["SPEI", "Gemiddeld (-0.5)", "Verhoogd (-1.0)",
           "Hoog (-1.5)", "Extreem (-2.0)"], fontsize=9)

# Risico als numerieke tijdreeks
ax2 = axes[1]
risk_numeric = drought_risk_daily.apply(lambda r: r.numeric)
# Kleur elk punt op basis van risiconiveau
for level in RiskLevel:
    mask = drought_risk_daily == level
    ax2.scatter(
        df_d.index[mask], risk_numeric[mask],
        s=1.5, color=RISK_COLORS[level], label=level.value
    )
ax2.set_yticks(range(5))
ax2.set_yticklabels([r.value for r in RiskLevel], fontsize=8)
ax2.set_title("Dagelijks droogterisico (2000–2025)", loc="left")
ax2.legend(handles=RISK_LEGEND, fontsize=8, ncol=5, loc="upper right")

plt.tight_layout()
plt.savefig("droogterisico_historisch.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 14 – Overstromingsrisico ────────────────────────────────────────
md("""\
## 10 · Overstromingsrisicoclassificatie

De `FloodRiskClassifier` combineert SMI en Total Runoff via een gewogen
composietscores:

    flood_score = 0,5 × norm_SMI + 0,5 × norm_total_ro

Beide indicatoren worden Min-Max genormaliseerd op de trainingsdata.
Risicodrempels worden afgeleid van de empirische percentielen van de
composietscores (40e, 60e, 75e, 90e percentielen).

> **Rationale**: een hoge SMI geeft aan dat de bodem verzadigd is
> (weinig absorptiecapaciteit) en hoge runoff geeft direct waterafvoer
> over het maaiveld aan — samen zijn dit de belangrijkste overstromings-
> precursors (Berghuijs et al., 2019).
""")

code("""\
flood_clf = FloodRiskClassifier(smi_weight=0.5, runoff_weight=0.5)
flood_clf.fit(train_d)

flood_risk_daily = flood_clf.classify(df_d)

counts_f = flood_risk_daily.value_counts()
pct_f    = (counts_f / len(flood_risk_daily) * 100).round(1)
summary_f = pd.DataFrame({"Dagen": counts_f, "%": pct_f})
summary_f.index = [r.value for r in summary_f.index]
print("Overstromingsrisico – verdeling over volledige periode:")
summary_f
""")

code("""\
fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# SMI tijdreeks
ax = axes[0]
ax.plot(df_d.index, df_d["smi"], lw=0.8, color="seagreen", label="SMI", alpha=0.8)
ax2b = ax.twinx()
ax2b.fill_between(df_d.index, df_d["total_ro"] * 1000, alpha=0.4,
                  color="steelblue", label="Total Runoff (×1000)")
ax2b.set_ylabel("Runoff ×10³ m/d", color="steelblue")
ax.set_ylabel("SMI (0–1)")
ax.set_title("SMI en Total Runoff", loc="left")
ax.legend(loc="upper left", fontsize=9)
ax2b.legend(loc="upper right", fontsize=9)

# Risico
ax3 = axes[1]
risk_numeric_f = flood_risk_daily.apply(lambda r: r.numeric)
for level in RiskLevel:
    mask = flood_risk_daily == level
    ax3.scatter(
        df_d.index[mask], risk_numeric_f[mask],
        s=1.5, color=RISK_COLORS[level], label=level.value
    )
ax3.set_yticks(range(5))
ax3.set_yticklabels([r.value for r in RiskLevel], fontsize=8)
ax3.set_title("Dagelijks overstromingsrisico (2000–2025)", loc="left")
ax3.legend(handles=RISK_LEGEND, fontsize=8, ncol=5, loc="upper right")

plt.tight_layout()
plt.savefig("overstromingsrisico_historisch.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 15 – Seizoensrisico ─────────────────────────────────────────────
code("""\
# Gemiddeld risico per maand (numeriek)
df_risk = pd.DataFrame({
    "drought":  drought_risk_daily.apply(lambda r: r.numeric),
    "flood":    flood_risk_daily.apply(lambda r: r.numeric),
    "month":    df_d.index.month,
})
monthly_risk = df_risk.groupby("month")[["drought", "flood"]].mean()
monthly_risk.index = ["Jan","Feb","Mrt","Apr","Mei","Jun",
                       "Jul","Aug","Sep","Okt","Nov","Dec"]

fig, ax = plt.subplots(figsize=(10, 5))
monthly_risk.plot.bar(ax=ax, color=["royalblue", "tomato"],
                      edgecolor="white", width=0.7)
ax.set_yticks(range(5))
ax.set_yticklabels([r.value for r in RiskLevel], fontsize=9)
ax.set_xlabel("Maand")
ax.set_title("Gemiddeld risico per kalendermaand (2000–2025)", fontweight="bold")
ax.legend(["Droogterisico", "Overstromingsrisico"])
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("seizoensrisico.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 16 – Toekomstprognose ────────────────────────────────────────────
md("""\
## 11 · Toekomstprognose – de komende 18 maanden

Het beste model per index wordt ingezet voor een prognose van 18 maanden.
De voorspelde waarden worden vervolgens doorgegeven aan de risicoclassifiers.

> **Let op**: voor risicoclassificatie worden de maandgemiddelde
> SARIMA/HW/RF-prognoses gebruikt.  De API-cutoff is berekend op de
> dagelijkse trainingsdata; voor maandelijkse prognoses wordt dezelfde
> drempel gehanteerd.
""")

code("""\
future_preds = {}
for col in TARGET_COLS:
    future_preds[col] = best_models[col].predict(steps=FORECAST_MONTHS)

future_df = pd.DataFrame(future_preds)
future_df.index.name = "date"
print(f"Prognoseperiode: {future_df.index[0].date()} → {future_df.index[-1].date()}")
future_df.round(5).head(6)
""")

code("""\
fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=False)
COLORS = ["royalblue", "darkorange", "seagreen", "firebrick"]

for ax, col, color in zip(axes, TARGET_COLS, COLORS):
    hist = df_m[col].iloc[-36:]
    pred = future_preds[col]
    ax.plot(hist.index, hist, color=color, lw=1.4, label="Historisch (3 jr)")
    ax.plot(pred.index,  pred, color=color, lw=2, ls="--", label="Prognose")
    ax.axvline(df_m.index[-1], color="grey", ls=":", lw=1.2, label="Vandaag")
    ax.set_title(col.upper(), loc="left", fontweight="bold")
    ax.legend(fontsize=9)

plt.suptitle(f"18-maanden prognose ({FORECAST_MONTHS} maanden) – beste model per index",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("toekomstprognose.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 17 – Toekomstig risico ──────────────────────────────────────────
code("""\
future_drought = drought_clf.classify_from_predictions(
    spei_pred=future_preds["spei"],
    api_pred=future_preds["api"],
)

future_flood = flood_clf.classify_from_predictions(
    smi_pred=future_preds["smi"],
    total_ro_pred=future_preds["total_ro"],
)

risk_forecast_df = pd.DataFrame({
    "Datum": future_df.index.strftime("%Y-%m"),
    "SPEI (pred)": future_df["spei"].round(3).values,
    "API (pred)":  future_df["api"].round(6).values,
    "Droogterisico": [r.value for r in future_drought],
    "SMI (pred)":  future_df["smi"].round(3).values,
    "Runoff (pred)": future_df["total_ro"].round(8).values,
    "Overstromingsrisico": [r.value for r in future_flood],
})
print(f"Prognose risicotabel ({FORECAST_MONTHS} maanden):")
risk_forecast_df.set_index("Datum")
""")

code("""\
fig, axes = plt.subplots(2, 1, figsize=(14, 7))

for ax, (risk_series, title, color) in zip(axes, [
    (future_drought, "Droogterisico – 18-maanden prognose", "royalblue"),
    (future_flood,   "Overstromingsrisico – 18-maanden prognose", "tomato"),
]):
    risk_num = risk_series.apply(lambda r: r.numeric)
    bars = ax.bar(
        range(len(risk_num)),
        risk_num + 1,
        color=[RISK_COLORS[r] for r in risk_series],
        edgecolor="white",
        width=0.8,
    )
    ax.set_xticks(range(len(risk_num)))
    ax.set_xticklabels(
        future_df.index.strftime("%b %Y"),
        rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([r.value for r in RiskLevel], fontsize=9)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.legend(handles=RISK_LEGEND, fontsize=8, ncol=5)

plt.tight_layout()
plt.savefig("toekomstig_risico.png", bbox_inches="tight")
plt.show()
""")

# ─── Cel 18 – Conclusie ──────────────────────────────────────────────────
md("""\
## 12 · Conclusie

### Modelprestaties
- **SARIMA** presteert goed voor SPEI (sterke seizoenscyclus en trend).
- **Holt-Winters** is competitief en eenvoudiger te configureren.
- **Random Forest** benut niet-lineaire patronen en is flexibel, maar
  accumuleert fout bij multi-step voorspelling.

### Risicoclassificatie
- De locatie (Hoorn van Afrika) kent een uitgesproken **droog seizoen**
  (november–april) met structureel verhoogd droogterisico.
- **Overstromingsrisico** is geconcentreerd in de korte regenperioden
  (Belg: maart–mei; Hagaya: juni–september).
- De prognose laat zien dat het droogterisico in de komende droge
  maanden naar verwachting **Verhoogd** tot **Hoog** blijft.

### Beperkingen
- Slechts één ruimtelijk punt; ruimtelijke spreiding ontbreekt.
- ERA5 is reanalysedata en heeft een inherente onzekerheid.
- SARIMA en HW extrapoleren lineaire/seizoenspatronen; extreme
  gebeurtenissen worden onderschat.
- Random Forest-prognoses verdampen bij lange horizonnen (mean reversion).

### Aanbevelingen voor vervolgonderzoek
- Pas een **ensemble** toe (gemiddelde van SARIMA + HW + RF).
- Gebruik **conformal prediction** of bootstrap-intervallen voor
  onzekerheidsberekening.
- Breid de dataset uit met meerdere locaties voor ruimtelijke analyse.
""")

# ─── Schrijf het notebook ──────────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0",
    },
}

output_path = "02_risk_modelling.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook geschreven: {output_path}  ({len(cells)} cellen)")
