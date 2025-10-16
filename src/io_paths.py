# src/io_paths.py
from pathlib import Path

# Dossier outputs à la racine du projet (ajuste si besoin)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR

def output_stem_for(path: Path) -> Path:
    """
    Retourne outputs/<stem> (sans suffixe). Exemple :
      input:  C:/.../take1_basic_pitch.mid  ->  outputs/take1_basic_pitch
    """
    ensure_outputs_dir()
    return OUTPUTS_DIR / path.stem
