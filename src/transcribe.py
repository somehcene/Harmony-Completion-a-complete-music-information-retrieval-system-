# ===============================
# File: src/transcribe.py
# ===============================
from pathlib import Path
import subprocess
from typing import Optional
from .config import (
    BASIC_PITCH_ONSET, BASIC_PITCH_FRAME, BASIC_PITCH_BENDS, BASIC_PITCH_TEMPO,
)


def basic_pitch_transcribe(
    audio_path: Path,
    out_dir: Path,
    onset: Optional[float] = BASIC_PITCH_ONSET,
    frame: Optional[float] = BASIC_PITCH_FRAME,
    bends: bool = BASIC_PITCH_BENDS,
    tempo: Optional[int] = BASIC_PITCH_TEMPO,
) -> Path:
    """Exécute la CLI Basic Pitch et renvoie le chemin du .mid généré.

    Note: Basic Pitch attend l'ordre <output_dir> puis <input_audio>.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = out_dir / f"{audio_path.stem}_basic_pitch.mid"

    cmd = ["basic-pitch"]
    if onset is not None:
        if not (0.0 <= onset <= 1.0):
            raise ValueError("onset must be in [0,1]")
        cmd += ["--onset-threshold", str(onset)]
    if frame is not None:
        if not (0.0 <= frame <= 1.0):
            raise ValueError("frame must be in [0,1]")
        cmd += ["--frame-threshold", str(frame)]
    if bends:
        cmd += ["--multiple-pitch-bends"]
    if tempo is not None:
        if tempo <= 0:
            raise ValueError("tempo must be > 0")
        cmd += ["--midi-tempo", str(tempo)]

    cmd += [str(out_dir), str(audio_path)]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            "basic-pitch CLI introuvable. Installe: pip install basic-pitch"
        )
    # fallback si le nom diffère
    if not midi_path.exists():
        candidates = sorted(out_dir.glob(f"{audio_path.stem}_basic_pitch*.mid"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            midi_path = candidates[0]
    if not midi_path.exists():
        raise RuntimeError("Aucun MIDI trouvé après exécution de Basic Pitch.")
    return midi_path