# ERA5 Time-Series Modellering en Risicoclassificatie
## Notebook uitleg, ontwerpkeuzes en bronvermelding

---

## Inhoud

1. [Doel en context](#1-doel-en-context)
2. [Dataset](#2-dataset)
3. [Architectuur en code-structuur](#3-architectuur-en-code-structuur)
4. [Data-voorbereiding](#4-data-voorbereiding)
5. [Stationariteit en seizoensanalyse](#5-stationariteit-en-seizoensanalyse)
6. [Keuze van tijdreeksmodellen](#6-keuze-van-tijdreeksmodellen)
   - [SARIMA](#61-sarima)
   - [Holt-Winters](#62-holt-winters-triple-exponential-smoothing)
   - [Random Forest](#63-random-forest-met-lag-features)
7. [Modelparameters en parameterinstelling](#7-modelparameters-en-parameterinstelling)
8. [Evaluatiemethode](#8-evaluatiemethode)
9. [Droogterisicoclassificatie](#9-droogterisicoclassificatie)
   - [SPEI als primaire indicator](#91-spei-als-primaire-indicator)
   - [API als modificator](#92-api-als-modificator)
10. [Overstromingsrisicoclassificatie](#10-overstromingsrisicoclassificatie)
    - [Composietscore](#101-composietscore-smi--total-runoff)
    - [Percentieldrempels](#102-percentieldrempels)
11. [Toekomstprognose](#11-toekomstprognose)
12. [Beperkingen](#12-beperkingen)
13. [Bronvermelding](#13-bronvermelding)

---

## 1. Doel en context

Het notebook `02_risk_modelling.ipynb` heeft als doel het voorspellen van vier
klimaatindices op basis van ERA5-reanalysedata en het omzetten van die
voorspellingen naar interpreteerbare risiconiveaus voor:

- **Droogte**: op basis van SPEI en API.
- **Overstroming**: op basis van SMI en Total Runoff.

De vijf risicoklassen zijn:

| Klasse | Omschrijving |
|--------|-------------|
| Laag risico | Normale of natte omstandigheden; geen directe actie vereist |
| Gemiddeld risico | Lichte afwijking; monitoring aanbevolen |
| Verhoogd risico | Significante afwijking; vroegtijdige waarschuwing |
| Hoog risico | Ernstige droogte of overstromingsdreiging; actie vereist |
| Extreem risico | Extreme gebeurtenis; noodsituatie mogelijk |

Het project past binnen het bredere **AI4AgriBizAfrica**-kader, gericht op
vroegtijdige waarschuwingssystemen voor droogte en overstroming in de Hoorn
van Afrika.

---

## 2. Dataset

| Kenmerk | Waarde |
|---------|--------|
| Bron | ERA5 reanalyse (ECMWF via Azure Blob Storage) |
| Locatie | 42,75°O / 9,25°N (Hoorn van Afrika, zuidwest-Ethiopië) |
| Periode | 2000-01-01 t/m 2025-12-31 |
| Frequentie | Dagelijks (9 497 rijen) |
| Doelvariabelen | `spei`, `api`, `smi`, `total_ro` |

### Variabelenbeschrijving

| Variabele | Eenheid | Beschrijving |
|-----------|---------|-------------|
| `spei` | – (gestandaardiseerd) | Standardized Precipitation-Evapotranspiration Index. Negatieve waarden duiden op droogte, positieve op natte omstandigheden. Warm-up periode van 29 dagen (NaN) aan het begin. |
| `api` | m | Antecedent Precipitation Index. Gewogen som van recentere neerslag; hoge waarden betekenen meer bodemvocht opgebouwd door voorafgaande neerslag. |
| `smi` | 0–1 | Soil Moisture Index. Genormaliseerde bodemvochtigheid (0 = volledig droog, 1 = volledig verzadigd). |
| `total_ro` | m/dag | Totale afvoer (opervlakkig + subsurface runoff). Sterk nul-geïnfleerd: >50 % van de dagwaarden is precies 0. |

---

## 3. Architectuur en code-structuur

Alle herbruikbare logica is geïmplementeerd in **`src/modeling.py`** volgens
OOP- en PEP8-principes. Het notebook importeert uitsluitend uit deze module.

```
src/modeling.py
├── RiskLevel                    # Enum: 5 risicoklassen
├── ModelMetrics                 # Dataclass: RMSE, MAE, R²
├── StationarityResult           # Dataclass: ADF-testresultaat
├── TimeSeriesPreprocessor       # Data laden, resamplen, splitsen
├── check_stationarity()         # ADF-test (module-level functie)
├── build_lag_features()         # Feature engineering voor RF
├── BaseForecaster (ABC)         # Abstracte basisklasse
│   ├── SARIMAForecaster         # Seasonal ARIMA
│   ├── HoltWintersForecaster    # Triple Exponential Smoothing
│   └── RandomForestForecaster   # Random Forest met lag-features
├── DroughtRiskClassifier        # SPEI + API → droogterisico
└── FloodRiskClassifier          # SMI + total_ro → overstromingsrisico
```

### Ontwerpkeuzes code

- **OOP**: elke modelklasse erft van `BaseForecaster` die de gedeelde
  `evaluate()`-methode levert. Dit voorkomt code-duplicatie en maakt het
  toevoegen van nieuwe modellen eenvoudig.
- **PEP8**: alle klassen en functies zijn voorzien van Google-stijl
  docstrings; import-volgorde is alphabetisch per sectie (stdlib → third-party
  → local); regellengtes ≤ 88 tekens (Black-standaard).
- **`fit` / `predict` interface**: geïnspireerd op de scikit-learn API
  (`fit` → `predict`). Dit maakt modellen uitwisselbaar.
- **Abstracte basisklasse**: `BaseForecaster` dwingt implementatie van
  `fit`, `predict` en `predict_in_sample` af via `@abstractmethod`.

---

## 4. Data-voorbereiding

### 4.1 Maandelijkse aggregatie

SARIMA en Holt-Winters werken op **maandelijkse** data (frequentie `'ME'`).
Redenen:

1. **Seizoensperiode**: een seizoensperiode van 365 (dagelijks) maakt SARIMA
   rekenintensief en numeriek onstabiel; `s=12` is standaard voor maandelijkse
   klimaatdata [Hyndman & Athanasopoulos, 2021].
2. **Nul-inflatie `total_ro`**: dagelijkse aggregatie naar maand vermindert
   de nul-inflatie van runoff significant.
3. **Rekentijd**: 300 maandelijkse observaties vs. 9 497 dagelijkse.

Aggregatiemethode per variabele:

| Type | Methode | Variabelen |
|------|---------|-----------|
| Cumulatief | `sum` | `tp`, `e`, `pev`, `total_ro`, `sro`, `lsp` |
| Gemiddeld | `mean` | `t2m`, `spei`, `api`, `smi`, `swvl1`, `swvl2` |

### 4.2 Missende waarden SPEI

De SPEI bevat 29 NaN-waarden aan het begin (warm-up van het berekeningsalgorithme).
Deze worden via **backward fill** (`bfill`) ingevuld met de vroegst beschikbare
waarde. Alternatief zou forward fill zijn, maar dat zou een toekomstige waarde
invullen op een historisch tijdstip; `bfill` is hier robuuster.

### 4.3 Train/test-splitsing

| Set | Periode | Maanden | Dagelijks |
|-----|---------|---------|-----------|
| Train | 2000-01 – 2022-12 | 276 | 8 401 |
| Test | 2023-01 – 2025-12 | 36 | 1 096 |

De splitsing is **temporeel** (niet willekeurig) om data-lekkage te voorkomen.

---

## 5. Stationariteit en seizoensanalyse

### 5.1 Augmented Dickey-Fuller test

De **Augmented Dickey-Fuller (ADF)** test toetst H₀: de tijdreeks heeft een
eenheidswortel (is non-stationair). Bij p < 0,05 wordt H₀ verworpen
(= stationair) [Dickey & Fuller, 1979].

Resultaten voor de maandelijkse indices (indicatief):

| Index | ADF p-waarde | Stationair |
|-------|-------------|-----------|
| SPEI | ~0,07 | Nee (d.w.z. `d=1` nodig in SARIMA) |
| API | ~0,03 | Ja (grenswaarde) |
| SMI | ~0,12 | Nee |
| Total Runoff | ~0,04 | Ja (grenswaarde) |

### 5.2 ACF / PACF

De auto- en partiële autocorrelatieplots tonen duidelijke pieken bij **lag 12**
voor SPEI en SMI, wat een jaarlijkse seizoenscyclus bevestigt. Dit motiveert
de keuze voor seizoensmodellen.

---

## 6. Keuze van tijdreeksmodellen

### 6.1 SARIMA

**SARIMA(p,d,q)(P,D,Q)s** is de uitbreiding van ARIMA met seizoenscomponenten
[Box et al., 2015]. Gekozen om:

- **Expliciete behandeling van seizoenaliteit** via de seizoensdifferentiatie
  `D=1` en seizoensauto-regressie `P`.
- **Stationariteitsvereiste** wordt opgelost via integratie `d`.
- **Breed gebruikt** in klimaatwetenschappelijke literatuur voor maandelijkse
  hydrologische tijdreeksen.

**Nadelen**: gevoelig voor parameterspecificatie, lineair van aard, langzaam
te trainen bij hogere ordes.

### 6.2 Holt-Winters (Triple Exponential Smoothing)

Holt-Winters past drie exponentiële afvlakkingen toe voor niveau (α),
trend (β) en seizoen (γ) [Winters, 1960]. Gekozen om:

- **Geen stationariteitseis**: werkt direct op de ruwe reeks.
- **Robuust bij kortere trainingsperioden** dan SARIMA.
- **Interpreteerbaar**: de componenten zijn expliciet zichtbaar.
- **Additief seizoen**: gekozen boven multiplicatief omdat de amplitude
  van de seizoensvariatie relatief stabiel is over de tijd.

**Nadelen**: extrapolerend (mag bij lange horizonnen niet vertrouwd worden),
gevoelig voor outliers in het seizoenscomponent.

### 6.3 Random Forest met lag-features

Random Forest [Breiman, 2001] transformeert het tijdreeksprobleem naar
gesuperviseerde regressie door de tijdreeks te voorzien van:

- **Lag-features**: t−1, t−2, t−3, t−6, t−12 (maanden)
- **Cyclische seizoensencoderingen**: `sin(2π·maand/12)` en `cos(2π·maand/12)`,
  alsmede dag-van-het-jaar codering

Gekozen om:

- **Niet-lineaire patronen**: kan complexe interacties tussen lags vastleggen.
- **Geen stationariteitseis**.
- **Feature-importantie**: geeft inzicht in welke lag het meest voorspellend is.
- **Multi-step**: via recursieve voorspelling (elk voorspeld datapunt wordt
  als lag-invoer gebruikt voor de volgende stap).

**Nadelen**: accumuleert fout bij multi-step voorspelling (recursief), neiging
naar mean-reversion bij lange horizonnen, geen intrinsieke onzekerheidsschatting.

---

## 7. Modelparameters en parameterinstelling

### SARIMA-parameters

| Index | (p,d,q) | (P,D,Q,s) | Motivatie |
|-------|---------|-----------|-----------|
| SPEI | (1,1,1) | (1,1,1,12) | Non-stationair (d=1), sterke seizoenspieken bij lag 12 (D=1, P=1, Q=1) |
| API | (1,0,1) | (1,1,1,12) | Near-stationair (d=0 sufficiënt), seizoenscomponent aanwezig |
| SMI | (1,1,1) | (1,1,1,12) | Vergelijkbaar met SPEI: seizoens-niet-stationair |
| Total Runoff | (1,1,0) | (0,1,1,12) | Hoog nul-aandeel; eenvoudige MA-structuur verkieslijk boven AR |

De parameterorde is bepaald op basis van:
1. ADF-testresultaten (keuze van `d`).
2. ACF/PACF-plots (keuze van `p`, `q`, `P`, `Q`).
3. Vuistregel: parsimonieuze modellen verkleinen overfitting.

In productie zou automatische selectie via **AIC-minimalisatie** (bijv.
`auto_arima` uit `pmdarima`) de voorkeur hebben.

### Holt-Winters-parameters

Alle indices: `trend='add'`, `seasonal='add'`, `seasonal_periods=12`.
De smoothingparameters (α, β, γ) worden automatisch geoptimaliseerd
via maximumlikelihood.

### Random Forest-parameters

| Parameter | Waarde | Motivatie |
|-----------|--------|-----------|
| `n_estimators` | 200 | Voldoende bomen voor stabiele schattingen; hoger geeft diminishing returns |
| `lags` | [1,2,3,6,12] | Dekt korte, mid- en lange termijngeheugen + jaarcyclus |
| `random_state` | 42 | Reproduceerbaarheid |

---

## 8. Evaluatiemethode

Modellen worden geëvalueerd op de **testset (2023–2025)** met drie metrics:

| Metric | Formule | Interpretatie |
|--------|---------|--------------|
| RMSE | √(Σ(y−ŷ)²/n) | Gevoelig voor grote fouten (uitschieters) |
| MAE | Σ|y−ŷ|/n | Robuust; gemiddelde absolute afwijking |
| R² | 1 − SS_res/SS_tot | Fractie variantie verklaard; 1 = perfect |

**Het beste model per index** (laagste RMSE op testset) wordt ingezet voor
de toekomstprognose.

---

## 9. Droogterisicoclassificatie

### 9.1 SPEI als primaire indicator

De SPEI is gestandaardiseerd zodat hij vergelijkbaar is over locaties en
tijdschalen. De drempelwaarden volgen de **McKee et al. (1993)**-classificatie:

| SPEI-bereik | Basisklasse |
|-------------|-------------|
| ≥ −0,5 | Laag risico |
| −1,0 tot −0,5 | Gemiddeld risico |
| −1,5 tot −1,0 | Verhoogd risico |
| −2,0 tot −1,5 | Hoog risico |
| < −2,0 | Extreem risico |

Deze drempels zijn **literatuur-gebaseerd** en wereldwijd geaccepteerd als
standaard voor SPEI-interpretatie [Vicente-Serrano et al., 2010].

### 9.2 API als modificator

De Antecedent Precipitation Index (API) weerspiegelt hoeveel neerslag er
in de voorafgaande periode is gevallen (gewogen naar recenticiteit)
[Kohler & Linsley, 1951]:

    API_t = k · API_{t-1} + P_t

waarbij `k` een afvlakkingscoëfficiënt is (typisch 0,85–0,98) en `P_t`
de dagelijkse neerslag.

**Waarom API als modificator?**  
Een lage API bij een al negatieve SPEI versterkt het droogtesignaal:
de bodem heeft zowel structureel (SPEI) als recent (API) weinig vocht
ontvangen.

**Implementatie**: als API < 25e percentiel van de trainingsdata EN
het SPEI-risiconiveau is niet EXTREEM, dan wordt het risiconiveau
één klasse verhoogd.

**Keuze 25e percentiel**: dit is een conservatieve drempel die alleen
de werkelijk droge gevallen markeert, waardoor false positives beperkt
blijven.

---

## 10. Overstromingsrisicoclassificatie

### 10.1 Composietscore (SMI + Total Runoff)

In tegenstelling tot SPEI heeft de SMI geen gestandaardiseerde
drempelwaarden in de literatuur (het is altijd lokaal gekalibreerd).
Daarom wordt een **composietscore** gebruikt:

    flood_score = 0,5 × norm_SMI + 0,5 × norm_total_ro

Beide indicatoren worden **Min-Max genormaliseerd** op de trainingsdata
zodat ze het bereik [0, 1] hebben.

**Rationale gewichten**: beide indicatoren hebben gelijke gewichten omdat
- SMI de verzadigingsgraad van de bodem meet (precondition voor overstroming)
- Total Runoff het directe overstromingssignaal is
- In de literatuur worden beide als even belangrijk beschouwd
  [Berghuijs et al., 2019].

### 10.2 Percentieldrempels

De risicodrempels worden afgeleid van de empirische percentielen van de
composietscores op de **trainingsdata**:

| Percentielbereik | Risicoklasse |
|-----------------|-------------|
| 0 – 40e | Laag risico |
| 40e – 60e | Gemiddeld risico |
| 60e – 75e | Verhoogd risico |
| 75e – 90e | Hoog risico |
| > 90e | Extreem risico |

**Waarom percentiel-gebaseerd?**  
- Geen universele drempelwaarden beschikbaar voor SMI.
- Percentieldrempels zijn adaptief aan de lokale klimaatverdeling.
- Het 90e percentiel voor "Hoog" correspondeert met 10 % van de dagen
  historisch overstromingsrisico, wat in lijn is met hydrologische
  praktijk [Vogel & Fennessey, 1994].

---

## 11. Toekomstprognose

Het beste model per index (laagste testset-RMSE) wordt gebruikt voor een
**18-maanden prognose**. De voorspelde SPEI en API worden doorgegeven aan
de `DroughtRiskClassifier`; de voorspelde SMI en Total Runoff aan de
`FloodRiskClassifier`.

**Beperkingen van de prognose**:
- SARIMA en Holt-Winters extrapoleren het geleerde seizoenspatroon maar
  kunnen geen abrupte klimaatanomalieën voorspellen.
- Random Forest convergeert op lange termijn naar het historisch gemiddelde
  (mean reversion door de recursieve lag-structuur).
- Er worden **geen betrouwbaarheidsintervallen** gerapporteerd. Voor
  productiesystemen zijn bootstrap-intervallen of conformal prediction
  aanbevolen.

---

## 12. Beperkingen

| Beperking | Toelichting |
|-----------|-------------|
| Eén ruimtelijk punt | Conclusies gelden alleen voor 42,75°O / 9,25°N; ruimtelijke heterogeniteit ontbreekt |
| ERA5 onzekerheid | Reanalysedata heeft een inherente modelonzekerheid (~10–20 % voor neerslag) |
| Lineaire seizoensaanname SARIMA/HW | Kan niet-lineaire klimaatveranderingssignalen missen |
| Nul-inflatie total_ro | De nul-inflatie beïnvloedt RMSE en maakt vergelijking tussen modellen moeilijker |
| Geen ensemble | Elk model heeft zijn zwakten; een ensemble zou robuuster zijn |
| API-definitie | API is hier berekend als een eenvoudige exponentiële afvlakking van dagelijkse neerslag; een verfijndere API-berekening met grondsoortparameters zou nauwkeuriger zijn |

---

## 13. Bronvermelding

### Methodologie tijdreeksmodellen

**Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).**
*Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
— Fundamentele referentie voor ARIMA en SARIMA.

**Winters, P. R. (1960).** Forecasting Sales by Exponentially Weighted
Moving Averages. *Management Science*, 6(3), 324–342.
https://doi.org/10.1287/mnsc.6.3.324
— Origineel artikel voor Holt-Winters exponential smoothing.

**Hyndman, R. J., & Athanasopoulos, G. (2021).**
*Forecasting: Principles and Practice* (3rd ed.). OTexts.
https://otexts.com/fpp3/
— Uitgebreid open-source leerboek; motivatie voor maandelijkse SARIMA
en Holt-Winters keuzes.

**Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32.
https://doi.org/10.1023/A:1010933404324
— Origineel Random Forest-artikel; basis voor de RF-implementatie.

### Droogte-indicatoren

**McKee, T. B., Doesken, N. J., & Kleist, J. (1993).** The relationship of
drought frequency and duration to time scales. *Preprints, 8th Conference
on Applied Climatology*, Anaheim, CA, Amer. Meteor. Soc., 179–184.
— Definitie van SPI-drempelwaarden; overgenomen voor SPEI.

**Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010).**
A Multiscalar Drought Index Sensitive to Global Warming: The Standardized
Precipitation Evapotranspiration Index. *Journal of Climate*, 23(7),
1696–1718. https://doi.org/10.1175/2009JCLI2909.1
— Definitie en interpretatie van SPEI.

**Kohler, M. A., & Linsley, R. K. (1951).** *Predicting the Runoff from
Storm Rainfall*. Research Note No. 34, U.S. Weather Bureau.
— Definitie van de Antecedent Precipitation Index (API).

### Overstromingsindicatoren

**Berghuijs, W. R., Harrigan, S., Molnar, P., Slater, L. J., & Kirchner,
J. W. (2019).** The relative importance of different flood-generating
mechanisms across Europe. *Water Resources Research*, 55(6), 4582–4593.
https://doi.org/10.1029/2019WR024841
— Ondersteunt het gebruik van bodemvocht en afvoer als overstromingsindicatoren.

**Martens, B., Miralles, D. G., Lievens, H., van der Schalie, R.,
de Jeu, R. A. M., et al. (2017).** GLEAM v3: satellite-based land
evaporation and root-zone soil moisture. *Geoscientific Model Development*,
10, 1903–1925. https://doi.org/10.5194/gmd-10-1903-2017
— Methodologie voor Soil Moisture Index (SMI).

**Vogel, R. M., & Fennessey, N. M. (1994).** Flow-Duration Curves I: New
Interpretation and Confidence Intervals. *Journal of Water Resources Planning
and Management*, 120(4), 485–504. https://doi.org/10.1061/(ASCE)0733-9496
— Basis voor percentielmethoden in hydrologische risicoclassificatie.

### Dataset en data-infrastructuur

**Hersbach, H., et al. (2020).** The ERA5 global reanalysis. *Quarterly
Journal of the Royal Meteorological Society*, 146(730), 1999–2049.
https://doi.org/10.1002/qj.3803
— Beschrijving van de ERA5-reanalysedata die als basis dient voor dit notebook.

**Dickey, D. A., & Fuller, W. A. (1979).** Distribution of the estimators
for autoregressive time series with a unit root. *Journal of the American
Statistical Association*, 74(366a), 427–431.
https://doi.org/10.1080/01621459.1979.10482531
— Theoretische basis voor de Augmented Dickey-Fuller stationariteitstest.

### Software

**Seabold, S., & Perktold, J. (2010).** Statsmodels: Econometric and
statistical modeling with Python. *Proceedings of the 9th Python in Science
Conference*, 57–61. https://doi.org/10.25080/Majora-92bf1922-011
— `statsmodels` bibliotheek voor SARIMA en Holt-Winters.

**Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python.
*Journal of Machine Learning Research*, 12, 2825–2830.
http://jmlr.org/papers/v12/pedregosa11a.html
— `scikit-learn` bibliotheek voor Random Forest en MinMaxScaler.

**The pandas development team (2024).** *pandas-dev/pandas: Pandas*.
Zenodo. https://doi.org/10.5281/zenodo.3509134
— Tijdreeksbeheer en resampling.
