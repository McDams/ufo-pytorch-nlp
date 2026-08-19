# UFO PyTorch NLP

Projet de Machine Learning Avancé : analyse de témoignages d'OVNI avec PyTorch.

La tâche principale consiste à prédire la forme observée (`shape`) à partir du
témoignage textuel (`comments`).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell : .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Exécution

```bash
python analyse.py
```

Les données sont téléchargées automatiquement dans `data/`.
Les résultats sont générés dans `outputs/`.