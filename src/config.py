# ===============================
# File: src/config.py
# ===============================
from pathlib import Path

# Dossiers par défaut
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "Data"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# Vocabulaire backend (CSV fourni)
BACKEND_VOCAB_CSV = DATA_DIR / "unique_chords.csv"

# Paramètres Basic Pitch (CLI)
BASIC_PITCH_ONSET  = None   # float in [0,1] or None
BASIC_PITCH_FRAME  = None   # float in [0,1] or None
BASIC_PITCH_BENDS  = False
BASIC_PITCH_TEMPO  = None   # int BPM or None

# Pré-traitement MIDI
MIN_NOTE_MS   = 60          # ignorer notes < 60 ms (80 si bruit)
DELTA_MERGE_MS= 25          # fusion d'onsets trop proches
QUANT_GRID    = "1/16"      # 1/16 ou 1/8
DEFAULT_TEMPO = 100         # si aucun tempo dans le MIDI

# Fenêtrage harmonique
WINDOW_MS     = 300         # 250–300 ms marche bien sans tempo
HOP_RATIO     = 0.5         # hop = WINDOW * HOP_RATIO

# Lissage
MEDIAN_K      = 3           # taille fenêtre filtre médian (impair)