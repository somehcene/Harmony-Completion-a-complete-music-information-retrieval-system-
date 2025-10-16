# ===============================
# File: src/midi_utils.py
# ===============================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable
from pathlib import Path
import music21 as m21
from .config import DEFAULT_TEMPO


@dataclass
class NoteEvent:
    onset: float   # seconds
    offset: float  # seconds
    pitch: int     # midi number
    velocity: int  # 0..127 (fallback 64 si inconnu)


def _estimate_tempo(stream: m21.stream.Stream) -> float:
    """Retourne un tempo BPM à partir des MetronomeMark, sinon DEFAULT_TEMPO."""
    mm = stream.metronomeMarkBoundaries()
    for (ts, te, mark) in mm:
        try:
            bpm = float(mark.number)
            if bpm > 0:
                return bpm
        except Exception:
            continue
    return float(DEFAULT_TEMPO)


def _ql_to_seconds(quarter_length: float, bpm: float) -> float:
    sec_per_beat = 60.0 / bpm
    return quarter_length * sec_per_beat


def read_midi_to_notes(midi_path: Path) -> List[NoteEvent]:
    s = m21.converter.parse(str(midi_path))
    bpm = _estimate_tempo(s)
    flat = s.flatten().notes
    notes: List[NoteEvent] = []
    for n in flat:
        if isinstance(n, m21.note.Note):
            onset_q = float(n.offset)
            dur_q = float(n.duration.quarterLength)
            onset_s = _ql_to_seconds(onset_q, bpm)
            offset_s = _ql_to_seconds(onset_q + dur_q, bpm)
            pitch = int(n.pitch.midi)
            vel = int(n.volume.velocity) if n.volume.velocity is not None else 64
            notes.append(NoteEvent(onset_s, offset_s, pitch, vel))
    return notes


def preprocess_notes(notes: List[NoteEvent], *, t_min_ms=60, vel_min=0,
                     delta_merge_ms=25, grid="1/16") -> List[NoteEvent]:
    """Filtrage durée/vélocité, fusion d'onsets proches, quantification simple."""
    import math

    def quant_step_seconds(grid: str) -> float:
        # grid comme '1/16' -> 0.25 beat; beat=quarter note
        den = int(grid.split("/")[-1])
        beat_frac = 4.0 / den
        return _ql_to_seconds(beat_frac, DEFAULT_TEMPO)

    tmin = t_min_ms / 1000.0
    dmerge = delta_merge_ms / 1000.0
    qstep = quant_step_seconds(grid)

    # 1) durée + vélocité
    keep = [n for n in notes if (n.offset - n.onset) >= tmin and n.velocity >= vel_min]
    keep.sort(key=lambda x: x.onset)

    # 2) fusion d'onsets proches (garde la note la plus longue)
    fused: List[NoteEvent] = []
    for n in keep:
        if not fused:
            fused.append(n); continue
        prev = fused[-1]
        if abs(n.onset - prev.onset) <= dmerge and n.pitch == prev.pitch:
            # garder la plus longue
            if (n.offset - n.onset) > (prev.offset - prev.onset):
                fused[-1] = n
        else:
            fused.append(n)

    # 3) quantification des onsets/offsets (arrondi au plus proche multiple)
    def q(x: float) -> float:
        return round(x / qstep) * qstep

    quanted = [NoteEvent(q(n.onset), q(n.offset), n.pitch, n.velocity) for n in fused]
    return quanted 
