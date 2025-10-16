# ===============================
# File: src/labeling.py
# ===============================
from typing import Optional, Set

NOTE_NAMES = {
    0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"
}

def intervals_from_root(root_pc: int, pcs: Set[int]) -> Set[int]:
    """Pitch-class set -> intervals above root (mod 12), excluding the root."""
    return {(p - root_pc) % 12 for p in pcs if p % 12 != root_pc % 12}

def has_third(I: Set[int]) -> bool:
    return 3 in I or 4 in I

def base_triad(I: Set[int]) -> Optional[str]:
    """Return '', 'm', 'dim', 'aug' for major/minor/dim/aug triads, else None."""
    has_m3, has_M3 = 3 in I, 4 in I
    has_P5, has_d5, has_A5 = 7 in I, 6 in I, 8 in I
    if has_M3 and has_P5:
        return ""          # major
    if has_m3 and has_P5:
        return "m"         # minor
    if has_M3 and has_A5:
        return "aug"
    if has_m3 and has_d5:
        return "dim"
    return None

def label_from_pcs(root_pc: int, pcs: Set[int], bass_pc: Optional[int] = None) -> str:
    """
    Heuristic chord labeling from pitch-classes:
    - triads (+ 6/7 extensions)
    - sus2 / sus4 (only if no 3rd present)
    - dyads (root+5th) -> root or slash if bass is fifth
    - power-chord '5' (not emitted explicitly: mapped to root)
    """
    if not pcs:
        return "UNK"

    root = NOTE_NAMES[root_pc]
    I = intervals_from_root(root_pc, pcs)

    # ---- dyads (2 notes) -----------------------------------------------------
    if len(pcs) == 2:
        # root + perfect fifth (or perfect fourth with inverted bass)
        if 7 in I or 5 in I:
            if bass_pc is not None and ((bass_pc - root_pc) % 12) in {7, 5}:
                return f"{root}/{NOTE_NAMES[bass_pc]}"
            return root  # treat as power chord
        # root + major/minor third -> major/minor
        if 4 in I:
            return root
        if 3 in I:
            return root + "m"
        # otherwise, default to root
        return root

    # ---- triad family --------------------------------------------------------
    triad = base_triad(I)

    # common upper extensions we support
    has_m7 = 10 in I
    has_M7 = 11 in I
    has_6  = 9  in I

    # sus2/sus4 only if there is NO third at all
    no_third = not has_third(I)
    has_2 = 2 in I
    has_4 = 5 in I
    has_5 = 7 in I

    # If no triad detected but it looks like a sus chord (2 or 4 with a 5)
    if triad is None and no_third and has_5:
        if has_2 and not has_4:
            return root + "sus2"
        if has_4 and not has_2:
            return root + "sus4"
        # both 2 and 4 (rare) -> pick sus4 by convention
        if has_2 and has_4:
            return root + "sus4"
        # fallback
        return root

    # Major triad
    if triad == "":
        # preference: M7 > 7 > 6 (mutually exclusive in typical pop cases)
        if has_M7:
            return root + "maj7"
        if has_m7:
            return root + "7"
        if has_6:
            return root + "6"
        # sus should not trigger when a 3rd is present
        return root

    # Minor triad
    if triad == "m":
        if has_m7:
            return root + "m7"
        if has_6:
            return root + "m6"
        # (m maj7) not emitted unless you want 'm(maj7)' in your vocab
        return root + "m"

    # Diminished
    if triad == "dim":
        # half-diminished (m7b5)
        if has_m7:
            return root + "m7b5"
        # fully diminished (dim7)
        if 9 in I:  # (technically double-flatted 7th -> 9 semitones above root)
            return root + "dim7"
        return root + "dim"

    # Augmented
    if triad == "aug":
        return root + "aug"

    # Fallback safe guard
    return root
