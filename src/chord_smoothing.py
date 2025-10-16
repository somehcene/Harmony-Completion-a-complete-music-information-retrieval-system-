# ===============================
# File: src/chord_smoothing.py
# ===============================
from statistics import mode
from dataclasses import dataclass
from typing import List

@dataclass
class ChordSeg:
    start: float  # seconds
    end: float    # seconds
    label: str

def collapse_repeats(labels: list[str]) -> list[str]:
    out = []
    for lab in labels:
        if not out or out[-1] != lab:
            out.append(lab)
    return out


def median_filter(labels: list[str], k: int = 3) -> list[str]:
    if k <= 1 or k % 2 == 0:
        return labels
    r = k // 2
    out = labels[:]
    for i in range(r, len(labels) - r):
        window = labels[i - r : i + r + 1]
        try:
            out[i] = mode(window)
        except Exception:
            out[i] = window[r]
    return out


def smooth(labels: list[str]) -> list[str]:
    return median_filter(collapse_repeats(labels), k=3)


def labels_to_segments(times: List[float], labels: List[str], hop_sec: float) -> List[ChordSeg]:
    """Convertit une grille (times, labels) en segments [start,end,label].
    On prend hop_sec comme granularité temporelle entre événements successifs.
    """
    assert len(times) == len(labels) and len(labels) > 0
    segs: List[ChordSeg] = []
    cur_label = labels[0]
    cur_start = times[0]
    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            segs.append(ChordSeg(start=cur_start, end=times[i], label=cur_label))
            cur_label = labels[i]
            cur_start = times[i]
    # dernier segment
    segs.append(ChordSeg(start=cur_start, end=times[-1] + hop_sec, label=cur_label))
    return segs


def enforce_min_duration(segs: List[ChordSeg], min_dur_sec: float) -> List[ChordSeg]:
    """Fusionne tout segment dont la durée < min_dur_sec dans le voisin le plus long.
    On répète jusqu'à ce qu'aucun segment ne soit sous le seuil.
    """
    if not segs:
        return []
    changed = True
    out = segs[:]
    while changed:
        changed = False
        if len(out) <= 1:
            break
        for i, s in enumerate(out):
            dur = s.end - s.start
            if dur >= min_dur_sec:
                continue
            # Choisir voisin pour fusion
            if i == 0:
                j = 1
            elif i == len(out) - 1:
                j = i - 1
            else:
                left_dur = out[i - 1].end - out[i - 1].start
                right_dur = out[i + 1].end - out[i + 1].start
                j = i - 1 if left_dur >= right_dur else i + 1
            # Fusion
            target = out[j]
            new_start = min(s.start, target.start)
            new_end = max(s.end, target.end)
            new_label = target.label
            lo, hi = (j, i) if j < i else (i, j)
            out.pop(hi)
            out.pop(lo)
            out.insert(lo, ChordSeg(new_start, new_end, new_label))
            changed = True
            break
    # Merge adjacents de même label
    merged: List[ChordSeg] = []
    for s in out:
        if merged and merged[-1].label == s.label:
            merged[-1].end = s.end
        else:
            merged.append(s)
    return merged
