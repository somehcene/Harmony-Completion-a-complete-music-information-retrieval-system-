# app_mic.py
from __future__ import annotations
import sys, io, json, math, uuid, datetime
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
from src.midi_utils import read_midi_to_notes  # pour le piano-roll

# ---- FFmpeg hint for pydub (Windows) ----
from pydub import AudioSegment
from pydub.utils import which
# essaie de trouver ffmpeg/ffprobe dans le PATH
ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")

# si non trouvés, mets un chemin absolu (À ADAPTER si besoin)
if not ffmpeg_path:
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"   # <- mets ton vrai chemin
if not ffprobe_path:
    ffprobe_path = r"C:\ffmpeg\bin\ffprobe.exe" # <- mets ton vrai chemin

AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

import os
if not (ffmpeg_path and Path(ffmpeg_path).exists()):
    st.error("FFmpeg introuvable. Installe-le (ou ajuste le chemin dans app_mic.py).")
    st.stop()

# ---- Rendre "src/" importable ----
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- Imports projet ----
from src.config import OUTPUTS_DIR, BACKEND_VOCAB_CSV
from src.transcribe import basic_pitch_transcribe
from src.cli import _analyze_midi
from src.backend_adapter import TrieSuggest
from src.io_paths import output_stem_for

# ---- Essayer le composant micro "streamlit-audiorecorder" ----
# pip install streamlit-audiorecorder pydub
try:
    from audiorecorder import audiorecorder  # renvoie un pydub.AudioSegment
    HAVE_AUDIOREC = True
except Exception:
    HAVE_AUDIOREC = False

# ==========================
# Helpers
# ==========================
def autoname(prefix: str = "take", ext: str = "wav") -> Path:
    """
    Crée un nom unique : outputs/take_YYYYmmdd-HHMMSS_<short>.wav
    """
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    fname = f"{prefix}_{ts}_{short}.{ext}"
    path = OUTPUTS_DIR / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

ROOT_TO_PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
def _root(s: str):
    if not s: return None
    return s[:2] if len(s)>1 and s[1] in "#b" else s[:1]
def _pc(s: str):
    r = _root(s); 
    return ROOT_TO_PC.get(r) if r else None
def _fifths(pc1: int, pc2: int) -> int:
    order = [0,7,2,9,4,11,6,1,8,3,10,5]
    i1, i2 = order.index(pc1), order.index(pc2)
    d = abs(i1 - i2)
    return min(d, 12 - d)
def tonal_score(prev: str|None, cand: str) -> float:
    if not prev: return 0.0
    p1, p2 = _pc(prev), _pc(cand)
    if p1 is None or p2 is None: return 0.0
    dist = _fifths(p1, p2)  # 0..6
    base = (6 - dist) / 6.0
    bonus = 0.0
    if (p1 - p2) % 12 == 7: bonus += 0.4  # V -> I
    if (p1 - p2) % 12 == 5: bonus += 0.2  # IV -> I
    if (p2 - p1) % 12 == 7: bonus += 0.15 # ii -> V (approx)
    if prev == cand:        bonus -= 0.2  # anti-répétition
    return base + bonus

# ==========================
# UI
# ==========================
st.set_page_config(page_title="Harmony Completion — Web Demo", layout="wide")
st.title("🎙️ Harmony Completion — Demo Web (micro + pipeline)")

with st.sidebar:
    st.header("Paramètres")
    onset  = st.slider("Onset threshold", 0.0, 1.0, 0.45, 0.01)
    frame  = st.slider("Frame threshold", 0.0, 1.0, 0.35, 0.01)
    tempo  = st.number_input("MIDI tempo", 30, 300, 96)
    min_ms = st.number_input("Durée minimale d’accord (ms)", 0, 3000, 600, step=50)
    tgt_rt = st.number_input("Taux cible (accords/s) — optionnel", 0.0, 3.0, 0.5, step=0.1)
    st.markdown("---")
    topk   = st.slider("Top-k suggestions", 1, 15, 8)
    alpha  = st.slider("α — poids Trie vs Tonal (ensemble)", 0.0, 1.0, 0.7, 0.05)
    mode   = st.radio("Mode de suggestion", ["Trie (corpus)", "Ensemble (Trie + Tonal)"])
    trie_pkl_path = OUTPUTS_DIR / "harmony_trie.pkl"
    st.caption(f"Trie: {trie_pkl_path}")

st.subheader("1) Audio : enregistre au micro ou uploade un fichier")

col1, col2 = st.columns([1,1])
audio_path: Path | None = None

# ---- A) Enregistrement micro ----
with col1:
    if HAVE_AUDIOREC:
        st.markdown("**Enregistrement micro**")
        st.caption("Clique sur Start, parle/joue, puis Stop. (navigateur requis)")
        audio_segment = audiorecorder("🎙️ Start recording", "🛑 Stop")
        if len(audio_segment) > 0:
            # audio_segment est un pydub.AudioSegment (PCM)
            out_wav = autoname(prefix="take", ext="wav")
            # Exporter en WAV
            buf = io.BytesIO()
            audio_segment.export(buf, format="wav")
            out_wav.write_bytes(buf.getvalue())
            audio_path = out_wav
            st.success(f"Enregistrement sauvegardé → {out_wav.name}")
            st.audio(str(out_wav))
    else:
        st.info("Module `streamlit-audiorecorder` non disponible. Active plutôt l'upload à droite.")

# ---- B) Upload fichier ----
with col2:
    st.markdown("**Upload audio** (.wav/.mp3)")
    uploaded = st.file_uploader("Dépose un fichier", type=["wav","mp3"])
    if uploaded is not None:
        out = autoname(prefix="upload", ext=Path(uploaded.name).suffix.lstrip(".").lower())
        out.write_bytes(uploaded.getbuffer())
        st.success(f"Upload sauvegardé → {out.name}")
        audio_path = out
        st.audio(str(out))

if audio_path is None:
    st.stop()

# ==========================
# 2) Transcription & Analyse
# ==========================
st.subheader("2) Transcription & Analyse")
if st.button("🚀 Transcrire & Analyser"):
    with st.spinner("Transcription Basic Pitch…"):
        midi_path = basic_pitch_transcribe(
            audio_path, OUTPUTS_DIR,
            onset=onset, frame=frame, bends=False, tempo=int(tempo)
        )
    st.success(f"MIDI généré → {midi_path.name}")

    with st.spinner("Analyse MIDI → accords normalisés…"):
        outstem = output_stem_for(audio_path)
        json_path = outstem.with_suffix(".chords.json")
        labels = _analyze_midi(
            midi_path,
            BACKEND_VOCAB_CSV,
            min_chord_ms=int(min_ms),
            target_rate=float(tgt_rt) if tgt_rt > 0 else None,
            json_out=json_path
        )
    st.success("Accords reconnus : " + " ".join(labels))

    # ---------- VISU : timeline d'accords + piano-roll ----------
st.markdown("### 🎼 Visualisations temporelles")

# 2.1 Chord timeline à partir du JSON {start,end,label} (en secondes)
if json_path.exists():
    data = json.loads(json_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)  # cols attendues: start (s), end (s), label (str)

    st.markdown("**Timeline des accords**")
    if not df.empty:
        # px.timeline attend des datetimes -> convertir secondes -> datetimes (origine Unix)
        df_tl = df.copy()
        df_tl["track"] = "Chords"
        df_tl["start_dt"] = pd.to_datetime(df_tl["start"], unit="s", origin="unix")
        df_tl["end_dt"]   = pd.to_datetime(df_tl["end"],   unit="s", origin="unix")

        import plotly.express as px
        fig = px.timeline(
            df_tl,
            x_start="start_dt", x_end="end_dt",
            y="track", color="label",
            hover_data={"label": True, "start": ":.3f", "end": ":.3f"},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            height=220, margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="Accord"
        )
        # Afficher l’axe en secondes (tickformat) même si c’est une échelle date
        fig.update_xaxes(title="Temps (s)", tickformat="%S.%L s")
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucun segment d’accord trouvé.")
else:
    df = pd.DataFrame({"label": labels})
    st.info("Aucun JSON temporel trouvé (pas de timeline).")

# 2.2 Piano-roll des notes extraites du MIDI (read_midi_to_notes)
try:
    # Lire les notes
    notes = read_midi_to_notes(midi_path)
    if notes:
        df_notes = pd.DataFrame(notes)

        # --- détection robuste des colonnes temps ---
        def pick(colnames, candidates):
            cl = [c.lower() for c in colnames]
            for cand in candidates:
                if cand in cl:
                    # renvoyer le vrai nom avec la casse d'origine
                    return colnames[cl.index(cand)]
            return None

        cols = list(df_notes.columns)
        start_col = pick(cols, ["start", "t0", "onset", "start_sec", "begin"])
        end_col   = pick(cols, ["end",   "t1", "offset","end_sec",   "finish"])
        pitch_col = pick(cols, ["pitch", "midipitch", "note", "midi"])

        # s’il manque quelque chose, on ne trace pas
        if not (start_col and end_col and pitch_col):
            st.info("Piano-roll indisponible (colonnes temps ou pitch absentes).")
        else:
            dfN = df_notes[[start_col, end_col, pitch_col]].rename(
                columns={start_col: "start", end_col: "end", pitch_col: "pitch"}
            ).copy()
            # garantir float
            dfN["start"] = pd.to_numeric(dfN["start"], errors="coerce")
            dfN["end"]   = pd.to_numeric(dfN["end"], errors="coerce")
            dfN = dfN.dropna(subset=["start", "end", "pitch"])
            dfN = dfN[dfN["end"] > dfN["start"]]

            if dfN.empty:
                st.info("Piano-roll : aucune note exploitable.")
            else:
                # Pitch -> nom lisible (C4, D#4,…)
                names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                def midi_name(p):
                    n = int(p)
                    return f"{names[n % 12]}{n//12 - 1}"
                dfN["name"] = dfN["pitch"].map(midi_name)

                st.markdown("**Piano-roll (notes MIDI)**")
                import altair as alt
                base = alt.Chart(dfN).mark_bar().encode(
                    x=alt.X("start:Q", title="Temps (s)"),
                    x2="end:Q",
                    y=alt.Y("name:N",
                            sort=alt.Sort(field="pitch", order="descending"),
                            title="Hauteur"),
                    tooltip=["name:N","pitch:Q",
                             alt.Tooltip("start:Q", format=".3f"),
                             alt.Tooltip("end:Q",   format=".3f")]
                ).properties(height=280)
                st.altair_chart(base, use_container_width=True)
    else:
        st.info("Aucune note MIDI lisible pour le piano-roll.")
except Exception as e:
    st.warning(f"Piano-roll non disponible : {e}")
# ---------- fin VISU ----------



    # ==========================
# 3) Suggestions
# ==========================
st.subheader("3) Suggestions")
ts = TrieSuggest.from_pkl(trie_pkl_path)

# Si aucune étiquette détectée, on arrête proprement
if not labels:
    st.warning("Aucune étiquette d’accord détectée. Lance d’abord la reconnaissance.")
    st.stop()

# Slider n-gram robuste (évite min==max)
max_n = min(6, len(labels))
if max_n <= 1:
    last_n = 1
    st.info("Contexte = 1 (un seul accord détecté) → n-gram fixé à 1.")
else:
    last_n = st.slider("Longueur de contexte (n-gram)", 1, max_n, min(3, max_n))

context = labels[-last_n:]
st.write(f"Contexte utilisé : `{context}`")

# Petit utilitaire de backoff (n → n-1 → ... → 1)
def trie_with_backoff(ts, ctx, k=8, debug=False):
    for n in range(len(ctx), 0, -1):
        sub = ctx[-n:]
        ranked = ts.top_next(sub, k=k, debug=debug) or []
        if ranked:
            return ranked, n, sub
    return [], 0, []

# ---- Mode Trie pur ----------------------------------------------------
if mode.startswith("Trie"):
    ranked, used_n, used_ctx = trie_with_backoff(ts, context, k=topk, debug=False)
    if not ranked:
        st.info("Trie : aucun match trouvé, même après backoff.")
    else:
        if used_n < len(context):
            st.caption(f"Backoff appliqué (n={used_n}) sur contexte : {used_ctx}")
        df2 = pd.DataFrame(ranked, columns=["candidat", "count"])
        st.dataframe(df2, use_container_width=True)

# ---- Mode Ensemble (Trie + Tonal) -------------------------------------
else:
    base, used_n, used_ctx = trie_with_backoff(ts, context, k=max(topk, 12), debug=False)
    if not base:
        st.info("Trie : aucun match trouvé, même après backoff → pas d’ensemble possible.")
    else:
        if used_n < len(context):
            st.caption(f"Backoff appliqué (n={used_n}) sur contexte : {used_ctx}")
        labels2, counts = zip(*base)

        # p_trie : softmax stable sur les counts
        m = max(counts)
        ex = [math.exp(c - m) for c in counts]
        s  = sum(ex) or 1.0
        p_trie = [e / s for e in ex]

        # p_ton : score tonal relatif au dernier accord du contexte utilisé
        prev = used_ctx[-1] if used_ctx else None
        p_ton = [tonal_score(prev, lab) for lab in labels2]

        # Fusion
        scored = []
        for lab, a, b in zip(labels2, p_trie, p_ton):
            score = alpha * a + (1.0 - alpha) * b
            scored.append((lab, score, a, b))
        scored.sort(key=lambda x: x[1], reverse=True)

        df3 = pd.DataFrame(scored[:topk], columns=["candidat", "score", "p_trie", "p_tonal"])
        st.dataframe(df3, use_container_width=True)

    # Sauvegarde accords texte
    out_txt = output_stem_for(audio_path).with_suffix(".chords.txt")
    out_txt.write_text(" ".join(labels), encoding="utf-8")
    st.caption(f"Accords sauvegardés → {out_txt}")
