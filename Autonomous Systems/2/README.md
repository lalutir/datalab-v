# Portfolio 2: Reinforcement Learning — Gymnasium Taxi-v3

## Overzicht

Dit project implementeert en vergelijkt twee Reinforcement Learning algoritmen — **Q-learning** en **SARSA** — in de [Gymnasium Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) omgeving. Een random-policy baseline dient als vergelijkingspunt. Alle algoritmen zijn volledig zelf geïmplementeerd, zonder gebruik te maken van bibliotheken die de algoritmen kant-en-klaar aanbieden (zoals Stable Baselines).

---

## Mappenstructuur

```
2/
├── main.ipynb                  # Hoofdnotebook: implementatie, experimenten, analyse
├── requirements.txt            # Alle benodigde Python-pakketten
├── README.md                   # Dit bestand
├── alpha_experiment.png        # Gegenereerd door het notebook: leersnelheid experiment
├── gamma_experiment.png        # Gegenereerd door het notebook: kortingsfactor experiment
├── epsilon_experiment.png      # Gegenereerd door het notebook: epsilon-verval experiment
├── comparison.png              # Gegenereerd door het notebook: Q-learning vs SARSA
├── performance_comparison.png  # Gegenereerd door het notebook: eindprestaties staafdiagram
└── q_table_heatmap.png         # Gegenereerd door het notebook: Q-tabel visualisatie
```

> De `.png`-bestanden worden aangemaakt wanneer het notebook volledig uitgevoerd wordt.

---

## Installatie en Gebruik

### Vereisten

- Python 3.10 of hoger
- pip

### Stap 1: Omgeving aanmaken (aanbevolen)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Stap 2: Pakketten installeren

```bash
pip install -r requirements.txt
```

### Stap 3: Notebook uitvoeren

Open het notebook in Jupyter en voer alle cellen van boven naar beneden uit:

```bash
jupyter notebook main.ipynb
```

Of gebruik VS Code met de Jupyter-extensie: open `main.ipynb` en klik op **Run All**.

---

## Inhoud van het Notebook

Het notebook is opgebouwd als een academisch rapport en bevat de volgende secties:

### 1. Probleemstelling
Beschrijving van de Taxi-v3 omgeving (staatruimte, actieruimte, beloningsstructuur) en een analyse van waarom Reinforcement Learning geschikt is voor dit probleem. Vergelijking met alternatieve methoden (supervised learning, regelgebaseerde systemen, zoekalgoritmen).

### 2. Theoretisch Kader
Uitleg van de basisconcepten van RL:
- **Staten, acties en beloningen** in de context van Taxi-v3
- **Q-learning** (off-policy TD-control): updateformule en werking
- **SARSA** (on-policy TD-control): verschil met Q-learning
- **Epsilon-greedy exploratie**: exploratie-exploitatie-afweging

### 3. Implementatie

#### 3.1 Setup
Importeren van bibliotheken, instellen van de random seed (42) voor reproduceerbaarheid, aanmaken van de omgeving.

#### 3.2 Baseline: Random Policy
Een willekeurige policy die bij elke stap een actie samplet. Evalueert over 1.000 episodes en rapporteert gemiddelde beloning, standaarddeviatie, gemiddeld aantal stappen en succespercentage.

#### 3.3 Q-learning Agent (`QLearningAgent`)
Klasse met de volgende methoden:
- `__init__`: Initialiseert de Q-tabel (500×6, gevuld met nullen) en hyperparameters
- `select_action(state)`: Epsilon-greedy actieselectie
- `update(state, action, reward, next_state, done)`: Off-policy Bellman-update: `Q(s,a) += α[r + γ·max Q(s',·) - Q(s,a)]`
- `decay_epsilon()`: Exponentieel verklein epsilon na elk episode
- `train(env, n_episodes, ...)`: Volledige trainingsloop
- `get_policy(state)`: Greedy actie (pure exploitatie)

#### 3.4 SARSA Agent (`SARSAAgent`)
Klasse met dezelfde interface als `QLearningAgent`, maar met een on-policy update:
- `update(state, action, reward, next_state, next_action, done)`: On-policy Bellman-update: `Q(s,a) += α[r + γ·Q(s',a') - Q(s,a)]`
- De trainingsloop kiest de volgende actie `a'` **vóór** de update (S,A,R,S',A' schema)

### 4. Hyperparameter Experimenten
Systematische sweep van drie hyperparameters met Q-learning (5.000 episodes per configuratie):

| Parameter | Geteste waarden | Standaardwaarde |
|-----------|----------------|-----------------|
| α (leersnelheid) | 0.01, 0.1, 0.5, 0.9 | 0.1 |
| γ (kortingsfactor) | 0.5, 0.8, 0.95, 0.99 | 0.99 |
| ε-verval | 0.990, 0.995, 0.999, 0.9995 | 0.995 |

Bevindingen:
- **α**: 0.1 is optimaal; te hoog (0.9) leidt tot instabiliteit, te laag (0.01) tot trage convergentie.
- **γ**: 0.99 is noodzakelijk; de finale beloning (+20) ligt ver in de toekomst en een lage γ maakt de agent bijziend.
- **ε-verval**: 0.995 biedt de beste balans; te snel verval beperkt exploratie, te langzaam vertraagt convergentie.

### 5. Resultaten en Analyse
- **Reward-curves**: voortschrijdend gemiddelde (venster=200) toont convergentie van Q-learning en SARSA ver boven de baseline.
- **Stappen per episode**: daalt sterk naarmate de policy verbetert.
- **Prestatie-overzicht**: tabel en staafdiagram met gemiddelde beloning, standaarddeviatie, stappen en succespercentage voor alle drie methoden.
- **Q-tabel heatmap**: visualiseert de geleerde Q-waarden; groene cellen zijn hoge (gewenste) waarden, rode cellen zijn lage waarden.

### 6. Discussie en Reflectie
- Analyse van Q-learning vs SARSA (off-policy vs on-policy)
- Beperkingen: tabellaire methode schaalt niet naar grotere omgevingen
- Toekomstig werk: Deep Q-Networks (DQN), Double Q-learning, Prioritized Experience Replay
- Conclusie

---

## Algoritmische Keuzes en Onderbouwing

### Waarom Q-learning en SARSA?
Beide algoritmen zijn klassieke, model-vrije TD-control methoden die goed passen bij Taxi-v3:
- De staatruimte is discreet en klein genoeg (500 staten) voor een tabellaire aanpak.
- Het algoritme vereist geen model van de omgevingsdynamica.
- Q-learning (Watkins & Dayan, 1992) en SARSA (Rummery & Niranjan, 1994) zijn fundamentele RL-algoritmen die de theoretische basis vormen voor modernere methoden.

### Waarom epsilon-greedy met exponentieel verval?
Epsilon-greedy is de meest gebruikte exploratiestrategie voor tabellaire RL. Exponentieel verval zorgt voor veel exploratie aan het begin (wanneer de Q-tabel nog leeg is) en steeds meer exploitatie naarmate de agent kennis opbouwt (Tokic, 2010).

### Waarom de Q-tabel initialiseren op nul?
Nul-initialisatie is een neutrale startpositie zonder aannames over de omgeving. Alternatief is optimistische initialisatie (hoge waarden), wat meer exploratie stimuleert, maar voor Taxi-v3 is nul-initialisatie voldoende.

### Reproduceerbare resultaten
Alle willekeurige elementen (Python `random`, NumPy) worden geïnitialiseerd met `SEED = 42`. Elke evaluatie-episode gebruikt `seed + episode` als seed voor de omgeving, zodat resultaten volledig reproduceerbaar zijn.

---

## Referenties

Bellman, R. (1957). A Markovian decision process. *Journal of Mathematics and Mechanics*, *6*(5), 679–684.

Dietterich, T. G. (2000). Hierarchical reinforcement learning with the MAXQ value function decomposition. *Journal of Artificial Intelligence Research*, *13*, 227–303.

Gymnasium. (2023). *Taxi — Gymnasium documentation*. Farama Foundation. https://gymnasium.farama.org/environments/toy_text/taxi/

Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, *518*(7540), 529–533.

Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems* (Technical Report CUED/F-INFENG/TR 166). University of Cambridge.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Tokic, M. (2010). Adaptive ε-greedy exploration in reinforcement learning based on value differences. In *KI 2010* (pp. 203–210). Springer.

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, *8*(3–4), 279–292.
