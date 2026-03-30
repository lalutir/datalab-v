# Autonomous Systems Notebook Setup

Deze README beschrijft hoe je het notebook in de map `Autonomous Systems Portfolio 1 Groep 3` kunt activeren.

## 📁 Bestandstructuur

- `Autonomous Systems Portfolio 1 Groep 3/main.ipynb` - belangrijkste notebook
- `Autonomous Systems Portfolio 1 Groep 3/requirements.txt` - Python dependencies

> Let op: de instructions in deze README gaan uit van het werken vanuit de folder `Autonomous Systems Portfolio 1 Groep 3`.

## 🚀 Stappen om het notebook te activeren

1. Open een terminal (PowerShell / WSL / Git Bash).
2. Ga naar de projectmap:

```powershell
cd "~/Autonomous Systems Portfolio 1 Groep 3"
```

3. Maak een nieuwe Python virtual environment aan (aanbevolen):

```powershell
python -m venv .venv
```

4. Activeer de virtual environment:
- PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- Command Prompt:

```cmd
.\.venv\Scripts\activate.bat
```

- Git Bash / WSL:

```bash
source .venv/Scripts/activate
```

5. Controleer dat de juiste Python interpreter actief is:

```powershell
python --version
```

6. Installeer dependencies uit `requirements.txt`:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

7. Start Jupyter Notebook / JupyterLab vanuit dezelfde omgeving:

```powershell
jupyter notebook
# of
jupyter lab
```

8. Open `main.ipynb` in je browser en voer de cellen uit.

## 🧹 Extra opschoon-stappen (optioneel)

- Om de virtuele omgeving uit te schakelen:

```powershell
deactivate
```

- Om op te ruimen:

```powershell
Remove-Item -Recurse -Force .venv
```

## 💡 Tips

- Gebruik altijd de virtuele omgeving voor dit project zodat systeem-breed geïnstalleerde Python packages geen conflicten geven.
- Als je met VS Code werkt:
  - open de projectmap (`File > Open Folder`) op `datalab-v` of `Autonomous Systems`
  - kies de interpreter `.venv\Scripts\python.exe`
  - gebruik `Jupyter: Run All Cells` in het notebook.

## 🔍 Probleemoplossing

- `ModuleNotFoundError`: controleer dat je in de correct geactiveerde venv zit.
- `pip` is oud: update eerst met `python -m pip install --upgrade pip`.
- `jupyter: command not found`: installeer `jupyter` via `pip install jupyter`.

---

Met deze stappen kun je het notebook veilig opstarten en werken in een gecontroleerde omgeving.