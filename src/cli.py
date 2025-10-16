# ===============================
# File: src/cli.py
# ===============================
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import json

from .config import (
    DATA_DIR, OUTPUTS_DIR, BACKEND_VOCAB_CSV, MIN_NOTE_MS, DELTA_MERGE_MS,
    QUANT_GRID, WINDOW_MS, HOP_RATIO, MEDIAN_K,
)
from .io_paths import output_stem_for
from .transcribe import basic_pitch_transcribe
from .midi_utils import read_midi_to_notes, preprocess_notes
from .chord_events import build_chord_events, choose_root
from .labeling import label_from_pcs
from .normalize import load_backend_vocab, normalize_to_backend
from .chord_smoothing import median_filter, labels_to_segments, enforce_min_duration, ChordSeg
from .acquisition import record_audio, RecConfig, ensure_wav


def cmd_acquire(args):
    audio_out = Path(args.out) if args.out else OUTPUTS_DIR / (args.name + ".wav")
    cfg = RecConfig(sr=args.sr, channels=args.channels, device=args.device)
    record_audio(audio_out, duration=args.duration, cfg=cfg)
    print("Audio:", audio_out)


def cmd_transcribe(args):
    midi_path = basic_pitch_transcribe(Path(args.audio), OUTPUTS_DIR,
                                       onset=args.onset, frame=args.frame,
                                       bends=args.bends, tempo=args.tempo)
    print("MIDI:", midi_path)


def _fix_unk_segments(segs: List[ChordSeg]) -> List[ChordSeg]:
    """
    Remplace les segments UNK par le label d'un voisin (priorité au précédent),
    puis refusionne les segments adjacents de même label.
    """
    if not segs:
        return segs
    out = [ChordSeg(s.start, s.end, s.label) for s in segs]
    n = len(out)
    for i, s in enumerate(out):
        if s.label == "UNK":
            repl = None
            if i > 0 and out[i-1].label != "UNK":
                repl = out[i-1].label
            elif i + 1 < n and out[i+1].label != "UNK":
                repl = out[i+1].label
            if repl:
                out[i].label = repl
    # fusionner les adjacents identiques
    merged: List[ChordSeg] = []
    for s in out:
        if merged and merged[-1].label == s.label:
            merged[-1].end = s.end
        else:
            merged.append(s)
    return merged


def _downsample_segments(segs: List[ChordSeg], target_rate: float | None) -> List[ChordSeg]:
    """
    Rééchantillonne grossièrement le nombre de segments vers ~ dur_total * target_rate.
    On conserve l’étiquette du premier segment du bloc.
    """
    if not target_rate or target_rate <= 0 or not segs:
        return segs
    total_dur = segs[-1].end - segs[0].start
    expected_n = int(round(total_dur * target_rate))
    if expected_n <= 0 or expected_n == len(segs):
        return segs

    factor = len(segs) / expected_n
    new_segs: List[ChordSeg] = []
    acc = 0.0
    buf: list[ChordSeg] = []
    for s in segs:
        buf.append(s)
        acc += 1
        if acc >= factor:
            merged = buf[0]
            merged_end = buf[-1].end
            new_segs.append(ChordSeg(start=merged.start, end=merged_end, label=merged.label))
            buf = []
            acc = 0.0
    if buf:
        merged = buf[0]
        merged_end = buf[-1].end
        new_segs.append(ChordSeg(start=merged.start, end=merged_end, label=merged.label))
    # fusion post-traitement si mêmes labels consécutifs
    merged2: List[ChordSeg] = []
    for s in new_segs:
        if merged2 and merged2[-1].label == s.label:
            merged2[-1].end = s.end
        else:
            merged2.append(s)
    return merged2


def _analyze_midi(midi_path: Path, vocab_csv: Path, *,
                  min_chord_ms: int | None = None,
                  target_rate: float | None = None,
                  json_out: Path | None = None) -> List[str]:
    # 1) Lecture + prétraitement notes
    notes = read_midi_to_notes(midi_path)
    notes = preprocess_notes(notes, t_min_ms=MIN_NOTE_MS,
                             delta_merge_ms=DELTA_MERGE_MS, grid=QUANT_GRID)

    # 2) Fenêtrage
    events = build_chord_events(notes, window_ms=WINDOW_MS, hop_ratio=HOP_RATIO)
    if not events:
        if json_out is not None:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text("[]", encoding="utf-8")
        return ["UNK"]

    times = [ev.time for ev in events]
    hop_sec = (WINDOW_MS / 1000.0) * HOP_RATIO

    # 3) Étiquetage brut (internes) — garder la longueur
    raw_labels = []
    for ev in events:
        r = choose_root(ev)
        raw = label_from_pcs(r, ev.pcs, ev.bass_pc)
        raw_labels.append(raw)

    # IMPORTANT : filtrage qui conserve la longueur
    raw_labels = median_filter(raw_labels, k=3)

    # 4) Normalisation vers vocab backend (longueur identique à times)
    vocab = load_backend_vocab(vocab_csv)
    mapped_full = [normalize_to_backend(l, vocab) for l in raw_labels]

    # 5) Segmentation (times x labels)
    segs = labels_to_segments(times, mapped_full, hop_sec)

    # 6) Nettoyage UNK -> voisin, puis fusion
    segs = _fix_unk_segments(segs)

    # 7) Durée minimale
    if min_chord_ms and min_chord_ms > 0:
        segs = enforce_min_duration(segs, min_chord_ms / 1000.0)

    # 8) Rééchantillonnage (optionnel)
    segs = _downsample_segments(segs, target_rate=target_rate)

    # 9) Export JSON (optionnel)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"start": round(s.start, 3), "end": round(s.end, 3), "label": s.label}
            for s in segs
        ]
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 10) Retour simple (liste des labels)
    return [s.label for s in segs]


def cmd_analyze(args):
    midi = Path(args.midi)
    outstem = output_stem_for(midi)
    json_path = outstem.with_suffix(".chords.json") if args.json else None
    labels = _analyze_midi(midi, Path(args.vocab),
                           min_chord_ms=args.min_chord_ms,
                           target_rate=args.target_rate,
                           json_out=json_path)
    out_txt = outstem.with_suffix(".chords.txt")
    out_txt.write_text(" ".join(labels), encoding="utf-8")
    print("Accords:", labels)
    print("Sauvé:", out_txt)
    if json_path:
        print("JSON:", json_path)


def cmd_full(args):
    audio = Path(args.audio)
    midi = basic_pitch_transcribe(audio, OUTPUTS_DIR,
                                  onset=args.onset, frame=args.frame,
                                  bends=args.bends, tempo=args.tempo)
    outstem = output_stem_for(audio)
    json_path = outstem.with_suffix(".chords.json") if args.json else None
    labels = _analyze_midi(midi, Path(args.vocab),
                           min_chord_ms=args.min_chord_ms,
                           target_rate=args.target_rate,
                           json_out=json_path)
    out_txt = outstem.with_suffix(".chords.txt")
    out_txt.write_text(" ".join(labels), encoding="utf-8")
    print("Accords:", labels)
    print("Sauvé:", out_txt)
    if json_path:
        print("JSON:", json_path)


def cmd_suggest_trie(args):
    # Backend trie
    from .backend_adapter import TrieSuggest
    from .normalize import load_backend_vocab

    # ===== Tonal fallback local (sans dépendance à src.tonal) =====
    ROOT_TO_PC = {
        "C":0, "C#":1, "Db":1, "D":2, "D#":3, "Eb":3, "E":4, "F":5,
        "F#":6, "Gb":6, "G":7, "G#":8, "Ab":8, "A":9, "A#":10, "Bb":10, "B":11
    }
    def _parse_root(ch: str):
        if not ch: return None
        return ch[:2] if len(ch)>1 and ch[1] in "#b" else ch[:1]
    def _parse_root_pc(ch: str):
        r = _parse_root(ch)
        return ROOT_TO_PC.get(r) if r else None
    def _circle_of_fifths_distance(pc1: int, pc2: int) -> int:
        order = [0,7,2,9,4,11,6,1,8,3,10,5]  # C G D A E B F# C# G# D# A# F
        i1, i2 = order.index(pc1), order.index(pc2)
        d = abs(i1 - i2)
        return min(d, 12 - d)
    def _cadence_bonus(prev: str|None, cand: str) -> float:
        if not prev: return 0.0
        p1 = _parse_root_pc(prev); p2 = _parse_root_pc(cand)
        if p1 is None or p2 is None: return 0.0
        if (p1 - p2) % 12 == 7:  # V -> I
            return 0.4
        if (p1 - p2) % 12 == 5:  # IV -> I
            return 0.2
        if (p2 - p1) % 12 == 7:  # ii -> V
            return 0.15
        return 0.0
    def _repetition_penalty(prev: str|None, cand: str) -> float:
        return -0.2 if prev and prev == cand else 0.0
    def _tonal_score(prev: str|None, cand: str) -> float:
        if not prev: return 0.0
        p1 = _parse_root_pc(prev); p2 = _parse_root_pc(cand)
        if p1 is None or p2 is None: return 0.0
        dist = _circle_of_fifths_distance(p1, p2)  # 0..6
        return (6 - dist) / 6.0
    def _rank_next_tonal(prefix: list[str], vocab: list[str], top_k: int = 5):
        prev = prefix[-1] if prefix else None
        scored = []
        for c in vocab:
            s = 0.35 * _tonal_score(prev, c)
            s += _cadence_bonus(prev, c)
            s += _repetition_penalty(prev, c)
            scored.append((c, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    # ===============================================================

    ts = TrieSuggest.from_pkl(args.trie_pkl)
    if args.seq:
        prefix = args.seq.strip().split()
    else:
        text = Path(args.from_file).read_text(encoding="utf-8")
        prefix = text.strip().split()

    out = ts.top_next(prefix, k=args.topk, debug=getattr(args, "debug", False))
    if not out:
        print("[suggest-trie] Aucun résultat pour ce préfixe (notation/corpus ?)")
        vocab = list(load_backend_vocab(BACKEND_VOCAB_CSV))
        tonal = _rank_next_tonal(prefix, vocab, top_k=args.topk)
        print("[fallback tonal] Propositions :")
        for lab, sc in tonal:
            print(f"{lab}\t{round(sc,4)}")
        return

    for ch, cnt in out:
        print(f"{ch}\t{cnt}")




def cmd_suggest(args):
    from .suggest import HarmonyCoreAdapter, NGramModel, rank_next, load_sequences_from_txt
    vocab_adapter = HarmonyCoreAdapter.from_csv_vocab(args.vocab)
    ngram = None
    if args.corpus:
        seqs = load_sequences_from_txt(args.corpus)
        if seqs:
            ngram = NGramModel(k=0.2); ngram.fit(seqs)
    if args.seq:
        prefix = args.seq.strip().split()
    else:
        text = Path(args.from_file).read_text(encoding="utf-8")
        prefix = text.strip().split()
    suggestions = rank_next(prefix, vocab_adapter, ngram=ngram, top_k=args.topk)
    for s in suggestions:
        print(f"{s.chord}\t{round(s.score,4)}\t{s.detail}")


def cmd_suggest_ensemble(args):
    from .backend_adapter import TrieSuggest
    # Tonal minimal inline (pas d'import externe)
    ROOT_TO_PC = {
        "C":0, "C#":1, "Db":1, "D":2, "D#":3, "Eb":3, "E":4, "F":5,
        "F#":6, "Gb":6, "G":7, "G#":8, "Ab":8, "A":9, "A#":10, "Bb":10, "B":11
    }
    def _root(s: str):
        if not s: return None
        return s[:2] if len(s)>1 and s[1] in "#b" else s[:1]
    def _pc(s: str):
        r = _root(s); 
        return ROOT_TO_PC.get(r) if r else None
    def _fifths(pc1: int, pc2: int) -> int:
        order = [0,7,2,9,4,11,6,1,8,3,10,5]
        i1, i2 = order.index(pc1), order.index(pc2)
        d = abs(i1 - i2); 
        return min(d, 12 - d)
    def tonal_score(prev: str|None, cand: str) -> float:
        if not prev: return 0.0
        p1, p2 = _pc(prev), _pc(cand)
        if p1 is None or p2 is None: return 0.0
        dist = _fifths(p1, p2)  # 0..6
        base = (6 - dist) / 6.0
        # bonus cadentiel léger
        bonus = 0.0
        if (p1 - p2) % 12 == 7: bonus += 0.4  # V->I
        if (p1 - p2) % 12 == 5: bonus += 0.2  # IV->I
        if (p2 - p1) % 12 == 7: bonus += 0.15 # ii->V
        if prev == cand:        bonus -= 0.2  # anti-répétition
        return base + bonus

    ts = TrieSuggest.from_pkl(args.trie_pkl)
    if args.seq:
        prefix = args.seq.strip().split()
    else:
        text = Path(args.from_file).read_text(encoding="utf-8")
        prefix = text.strip().split()

    ranked = ts.ensemble_rank(prefix, k=args.topk, tonal_fn=tonal_score, alpha=args.alpha, debug=getattr(args, "debug", False))
    if not ranked:
        print("[ensemble] Trie vide pour ce préfixe → essaye 'suggest' (tonal) ou 'suggest-trie --debug'")
        return
    for lab, sc, detail in ranked:
        print(f"{lab}\t{round(sc,4)}\t{detail}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("HCProject pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Acquire ------------------------------------------------------------
    p0 = sub.add_parser("acquire", help="Enregistrer de l'audio depuis le micro")
    p0.add_argument("--name", default="take")
    p0.add_argument("--out", help="Chemin WAV de sortie (sinon outputs/<name>.wav)")
    p0.add_argument("--duration", type=float, help="Durée en secondes (sinon Ctrl+C pour stop)")
    p0.add_argument("--sr", type=int, default=44100)
    p0.add_argument("--channels", type=int, default=1)
    p0.add_argument("--device", help="ID/Nom périphérique audio (facultatif)")
    p0.set_defaults(func=cmd_acquire)

    # Transcribe ---------------------------------------------------------
    p1 = sub.add_parser("transcribe", help="Audio -> MIDI via Basic Pitch")
    p1.add_argument("audio")
    p1.add_argument("--onset", type=float)
    p1.add_argument("--frame", type=float)
    p1.add_argument("--bends", action="store_true")
    p1.add_argument("--tempo", type=int)
    p1.set_defaults(func=cmd_transcribe)

    # Analyze ------------------------------------------------------------
    p2 = sub.add_parser("analyze", help="MIDI -> accords (normalisés backend)")
    p2.add_argument("midi")
    p2.add_argument("--vocab", default=str(BACKEND_VOCAB_CSV))
    p2.add_argument("--min-chord-ms", type=int, default=500,
                    help="Durée minimale d'un segment d'accord (fusion si plus court)")
    p2.add_argument("--target-rate", type=float, default=None,
                    help="Taux approximatif d'accords par seconde (ex: 0.5 ≈ 3 accords / 6s)")
    p2.add_argument("--json", action="store_true", help="Exporter un JSON {start,end,label}")
    p2.set_defaults(func=cmd_analyze)

    # Full ---------------------------------------------------------------
    p3 = sub.add_parser("full", help="Audio -> MIDI -> accords")
    p3.add_argument("audio")
    p3.add_argument("--onset", type=float)
    p3.add_argument("--frame", type=float)
    p3.add_argument("--bends", action="store_true")
    p3.add_argument("--tempo", type=int)
    p3.add_argument("--vocab", default=str(BACKEND_VOCAB_CSV))
    p3.add_argument("--min-chord-ms", type=int, default=500)
    p3.add_argument("--target-rate", type=float, default=None)
    p3.add_argument("--json", action="store_true")
    p3.set_defaults(func=cmd_full)

    # suggest parser
    p4 = sub.add_parser("suggest", help="Proposer les prochains accords")
    src = p4.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", help="Séquence d'entrée, ex: 'C G Am F'")
    src.add_argument("--from-file", help="Chemin vers un .chords.txt")
    p4.add_argument("--topk", type=int, default=5)
    p4.add_argument("--vocab", default=str(BACKEND_VOCAB_CSV))
    p4.add_argument("--corpus", nargs="*", help="Liste de .chords.txt pour entraîner le bigramme")
    p4.set_defaults(func=cmd_suggest)

    p5 = sub.add_parser("suggest-trie", help="Suggestions depuis harmony_trie.pkl")
    src = p5.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", help="Séquence d'entrée, ex: 'C G Am'")
    src.add_argument("--from-file", help="Chemin vers un .chords.txt")
    p5.add_argument("--topk", type=int, default=5)
    p5.add_argument("--trie-pkl", default=str(OUTPUTS_DIR / "harmony_trie.pkl"))
    p5.add_argument("--debug", action="store_true")
    p5.set_defaults(func=cmd_suggest_trie)


    p6 = sub.add_parser("suggest-ensemble", help="Trie + tonal (ensemble)")
    src6 = p6.add_mutually_exclusive_group(required=True)
    src6.add_argument("--seq", help="Séquence d'entrée, ex: 'F# B/F# F#'")
    src6.add_argument("--from-file", help="Chemin vers un .chords.txt")
    p6.add_argument("--topk", type=int, default=5)
    p6.add_argument("--trie-pkl", default=str(OUTPUTS_DIR / "harmony_trie.pkl"))
    p6.add_argument("--alpha", type=float, default=0.7, help="poids trie vs tonal (0..1)")
    p6.add_argument("--debug", action="store_true")
    p6.set_defaults(func=cmd_suggest_ensemble)
    
    return p


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
