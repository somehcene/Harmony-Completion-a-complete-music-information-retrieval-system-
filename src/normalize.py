# ===============================
# File: src/normalize.py
# ===============================
from pathlib import Path
import re

# prefer sharps to match your backend
ENHARMONIC_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

NOTE_RE = r"[A-G](?:#|b)?"

def prefer_sharps(n: str) -> str:
    return ENHARMONIC_TO_SHARP.get(n, n)

def load_backend_vocab(csv_path: Path) -> set[str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "chord" not in df.columns:
        df = df.rename(columns={df.columns[0]: "chord"})
    return set(df["chord"].dropna().astype(str).str.strip().unique())

# body recognizes usual keywords; we keep them separate from the root
LABEL_RX = re.compile(
    fr"^(?P<root>{NOTE_RE})"
    r"(?P<body>(maj7|m7b5|dim7|maj|m6|m7|aug|dim|sus2|sus4|6|7|m)?)"
    fr"(?:/(?P<bass>{NOTE_RE}))?$"
)

def candidate_strings(root: str, triad: str, ext: str | None, bass: str | None) -> list[str]:
    r = prefer_sharps(root)
    b = prefer_sharps(bass) if bass else None
    base = r if triad == "" else r + triad
    if ext:
        base = base + ext
    with_slash = f"{base}/{b}" if b else None
    return [x for x in [with_slash, base] if x]

def normalize_to_backend(raw_label: str, backend_vocab: set[str]) -> str:
    """
    Normalize a raw chord label (from labeling.py) into a form present
    in the backend vocabulary (unique_chords.csv).
    """
    if not raw_label or raw_label == "UNK":
        return "UNK"

    # sanitization
    L = (
        raw_label.replace("–", "-")
        .replace("—", "-")
        .replace(" ", "")
        .replace("_", "")
        .replace("other", "")
    )

    # --------- tolerant word-based normalization (case-insensitive) ----------
    # map 'Major', 'major', 'maj' (as a triad marker) to nothing,
    # and seven types to canonical tokens
    # Keep the root's case as-is; apply to suffix part only.
    # Accept forms like C:major, C-major, Cmaj, Cmajorseventh, CM7, etc.
    # (We only touch the tail of the string.)

    # Longest patterns first to avoid breaking 'maj7'
    L = re.sub(r"(?i)majorseventh$", "maj7", L)
    L = re.sub(r"(?i)dominantseventh$", "7", L)
    L = re.sub(r"(?i)minorseventh$", "m7", L)
    L = re.sub(r"(?i)mmaj7$", "m(maj7)", L)  # keep if ever encountered

    # Triad words at the end (C:major, C-major, Cmajor -> C ; C:minor -> Cm)
    L = re.sub(rf"^({NOTE_RE})(?:\:|-)?(?i:major)$", r"\1", L)
    L = re.sub(rf"^({NOTE_RE})(?:\:|-)?(?i:minor|min)$", r"\1m", L)

    # Compact synonyms: CM7 -> Cmaj7; Cm7 already ok; Cmin7 -> Cm7
    L = re.sub(rf"^({NOTE_RE})M7$", r"\1maj7", L)
    L = re.sub(rf"^({NOTE_RE})(?i:min7)$", r"\1m7", L)

    # Allow bare 'maj' after the root: Cmaj -> C
    L = re.sub(rf"^({NOTE_RE})(?i:maj)$", r"\1", L)

    # -------------------------------------------------------------------------
    m = LABEL_RX.fullmatch(L)
    if not m:
        return "UNK"

    root = m.group("root")
    body = m.group("body") or ""
    bass = m.group("bass")

    # In our grammar:
    #  - triad part is one of: '', 'm', 'dim', 'aug', 'maj'
    #  - extension part is one of: 6, 7, maj7, m6, m7, m7b5, dim7, sus2, sus4
    triad_map = {
        "": "",
        "m": "m",
        "dim": "dim",
        "aug": "aug",
        "maj": "",  # treat 'maj' as plain major triad
    }
    ext_set = {"6", "7", "maj7", "m6", "m7", "m7b5", "dim7", "sus2", "sus4"}

    triad = body if body in triad_map else ""
    ext = body if body in ext_set else None
    triad = triad_map.get(triad, "")

    # Try (with bass) -> (no bass) fallbacks
    for s in candidate_strings(root, triad, ext, bass):
        if s in backend_vocab:
            return s
    for s in candidate_strings(root, triad, ext, None):
        if s in backend_vocab:
            return s
    for s in candidate_strings(root, triad, None, bass):
        if s in backend_vocab:
            return s
    for s in candidate_strings(root, triad, None, None):
        if s in backend_vocab:
            return s
    # Finally, try plain triad or just the root
    for s in candidate_strings(root, "", None, None):
        if s in backend_vocab:
            return s

    return "UNK"
