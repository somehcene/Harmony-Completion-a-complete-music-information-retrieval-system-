# ===============================
# File: src/acquisition.py
# ===============================
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Acquisition micro via sounddevice + soundfile (cross-platform)
# pip install sounddevice soundfile

@dataclass
class RecConfig:
    sr: int = 44100
    channels: int = 1
    device: Optional[int | str] = None  # None = default
    dtype: str = "float32"


def record_audio(out_path: Path, duration: Optional[float] = None, cfg: RecConfig = RecConfig()) -> Path:
    """Enregistre un fichier WAV au chemin donné.
    - duration en secondes ; si None => enregistrement jusqu'à Ctrl+C.
    - nécessite les paquets: sounddevice, soundfile.
    Retourne out_path.
    """
    import sounddevice as sd
    import soundfile as sf
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if duration is not None and duration <= 0:
        raise ValueError("duration must be > 0 or None")

    sd.default.samplerate = cfg.sr
    sd.default.channels = cfg.channels
    if cfg.device is not None:
        sd.default.device = cfg.device

    if duration is None:
        # Streaming jusqu'à Ctrl+C
        print("[REC] Recording... press Ctrl+C to stop")
        try:
            with sf.SoundFile(str(out_path), mode='w', samplerate=cfg.sr, channels=cfg.channels, subtype='PCM_16') as f:
                with sd.InputStream(dtype=cfg.dtype, callback=lambda indata, frames, time, status: f.write(indata)):
                    while True:
                        sd.sleep(250)
        except KeyboardInterrupt:
            print("[REC] Stopped.")
    else:
        print(f"[REC] Recording {duration}s ...")
        audio = sd.rec(int(duration * cfg.sr), samplerate=cfg.sr, channels=cfg.channels, dtype=cfg.dtype)
        sd.wait()
        sf.write(str(out_path), audio, cfg.sr, subtype='PCM_16')
        print("[REC] Saved", out_path)

    return out_path


def list_devices() -> None:
    """Affiche les périphériques audio disponibles."""
    import sounddevice as sd 
    print(sd.query_devices())


def ensure_wav(input_path: Path, out_path: Optional[Path] = None, sr: int = 44100) -> Path:
    """Convertit n'importe quel audio en WAV mono sr donné (ffmpeg requis)."""
    import subprocess
    from shutil import which
    if which('ffmpeg') is None:
        raise RuntimeError("ffmpeg requis pour la conversion (installez-le)")
    if out_path is None:
        out_path = input_path.with_suffix('.wav')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-ac', '1', '-ar', str(sr), str(out_path)
    ]
    subprocess.run(cmd, check=True)
    return out_path