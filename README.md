# 🎵 Harmony Completion: a Complete Music Information Retrieval System

> **Harmony Completion** is a compact end-to-end **Music Information Retrieval (MIR)** system designed to **analyze, normalize, and suggest musical chords** from raw audio.  
> Built during a research internship at *LIP6 – Sorbonne Université*, it explores **algorithmic approaches** for harmony completion inspired by **Information Retrieval** and **music theory**.

## 🌐 Overview
Harmony Completion bridges the gap between **audio processing**, **symbolic analysis**, and **musical reasoning**.  
It integrates all stages of the MIR pipeline — from recording to harmonic suggestion — inside a unified framework.

### 🧩 Pipeline
1. **Audio Acquisition** → real-time microphone recording (`streamlit` or CLI)
2. **Transcription** → `Basic Pitch` for audio-to-MIDI conversion
3. **MIDI Analysis** → symbolic note grouping and chord inference
4. **Normalization** → mapping chords to a backend vocabulary (`unique_chords.csv`)
5. **Smoothing & Filtering** → removes noise, merges micro-events
6. **Suggestion Engine** → next-chord prediction via:
   - a **Trie model** (frequency-based)
   - a **Tonal model** (circle-of-fifths distance + cadence heuristics)
   - an **Ensemble model** combining both (α-weighted fusion)

## ⚙️ System Architecture
```
src/
├── acquisition.py
├── transcribe.py
├── midi_utils.py
├── labeling.py
├── chord_smoothing.py
├── normalize.py
├── backend_adapter.py
├── tonal.py
├── cli.py
├── app_mic.py
└── io_paths.py, config.py, etc.
```

## 🧠 Key Ideas
### 🔍 Harmony as Information Retrieval
Predict the next chord by ranking candidates using:
- **Corpus likelihood** (via trie)
- **Tonal compatibility** (via music theory)
- **α-fusion**: balance between learned style and harmonic logic

### 🎼 Tonal Distance Function
```
tonal(prev, cand) = (6 - fifths_distance(prev, cand))/6 + cadence_bonus(prev, cand) - repetition_penalty
```

## 🧑‍💻 Usage
### CLI
```
python -m src.cli acquire --name demo_take --duration 5
python -m src.cli full outputs/demo_take.wav --onset 0.45 --frame 0.35 --tempo 96
python -m src.cli suggest-ensemble --from-file outputs/demo_take.chords.txt --topk 8 --alpha 0.7
```

### Web App
```
streamlit run app_mic.py
```

## 📊 Data & Training
- **Dataset:** [MusicBench](https://github.com/mir-dataset/MusicBench)
- **Vocabulary:** `unique_chords.csv` (~157 chords)
- **Trie Construction:** frequency-based prefix tree built from corpus

## 📚 References
- Lerdahl, *Tonal Pitch Space* (2001)
- Temperley, *Cognition of Basic Musical Structures* (2001)
- Spotify Basic Pitch
- MusicBench (2024)

## 🧑‍🔬 Author
**Ahcene Loubar**  
M1 RES – Sorbonne Université  
🎓 Research Internship @ LIP6 – Networks and Performance Analysis (NPA)  
📧 ahcene.loubar@etu.sorbonne-universite.fr
