# src/notation.py
from __future__ import annotations
from typing import List, Tuple

SHARP_TO_FLAT = {"C#":"Db","D#":"Eb","F#":"Gb","G#":"Ab","A#":"Bb"}
FLAT_TO_SHARP = {v:k for k,v in SHARP_TO_FLAT.items()}

def enharmonics(n: str) -> List[str]:
    # retourne [n, alt] si #/b; sinon [n]
    if n in SHARP_TO_FLAT: return [n, SHARP_TO_FLAT[n]]
    if n in FLAT_TO_SHARP: return [n, FLAT_TO_SHARP[n]]
    return [n]

def split_chord(ch: str) -> Tuple[str, str|None]:
    # "B/F#" -> ("B", "F#") ; "F#m7" -> ("F#", None)
    if "/" in ch:
        root, bass = ch.split("/", 1)
        return root, bass
    return ch, None

def join_chord(root: str, bass: str|None) -> str:
    return f"{root}/{bass}" if bass else root

def variants_for_chord(ch: str) -> List[str]:
    """
    Génère des variantes probables pour matcher le corpus :
      - conserver tel quel,
      - enharmoniques root et basse (ex: B/F# -> B/Gb),
      - retirer l'inversion (B/F# -> B),
      - (option) maj/min marker neutre si déjà une triade simple.
    """
    root, bass = split_chord(ch)
    outs: list[str] = []
    roots = enharmonics(root)
    basses = enharmonics(bass) if bass else [None]
    for r in roots:
        for b in basses:
            outs.append(join_chord(r, b))
    # fallback sans inversion
    for r in roots:
        outs.append(r)
    # dédoublonner en gardant l'ordre
    seen = set(); uniq = []
    for o in outs:
        if o not in seen:
            seen.add(o); uniq.append(o)
    return uniq
