# ===============================
# File: src/chord_events.py
# ===============================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple
from .midi_utils import NoteEvent
import music21 as m21


@dataclass
class ChordEvent:
    time: float                 # début fenêtre (s)
    pcs: Set[int]               # pitch classes actifs dans la fenêtre
    bass_pc: int | None         # pitch-class de basse (si dispo)
    pitches: List[int]          # toutes les hauteurs (midi) rencontrées


def _active_in_window(n: NoteEvent, t0: float, t1: float) -> bool:
    return not (n.offset <= t0 or n.onset >= t1)


def build_chord_events(notes: List[NoteEvent], window_ms=300, hop_ratio=0.5) -> List[ChordEvent]:
    if not notes:
        return []
    tmin = min(n.onset for n in notes)
    tmax = max(n.offset for n in notes)
    win = window_ms / 1000.0
    hop = win * hop_ratio

    t = tmin
    events: List[ChordEvent] = []
    while t <= tmax:
        t0, t1 = t, t + win
        actives = [n for n in notes if _active_in_window(n, t0, t1)]
        if actives:
            pcs = {p % 12 for p in [n.pitch for n in actives]}
            # basse: note la plus grave au plus proche de t0
            actives.sort(key=lambda n: (n.onset, n.pitch))
            bass_pc = (min(actives, key=lambda n: n.pitch).pitch) % 12 if actives else None
            ev = ChordEvent(time=t0, pcs=pcs, bass_pc=bass_pc, pitches=[n.pitch for n in actives])
            events.append(ev)
        t += hop
    return events


def choose_root(event: ChordEvent) -> int:
    """Essaie music21.Chord.root(), fallback sur la basse."""
    try:
        chord = m21.chord.Chord(event.pitches)
        r = chord.root().pitchClass
        return int(r)
    except Exception:
        pass
    return int(event.bass_pc) if event.bass_pc is not None else min(event.pcs)

