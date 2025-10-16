# === PATCH build_trie.py ===
from .trie import HarmonyTrie
from .loader import load_musicbench_jsonl

def build_harmony_trie(entries):
    trie = HarmonyTrie()
    for entry in entries:
        chords = entry.get("chords", [])
        if chords and len(chords) > 1:
            # on insère TOUTES les prefixes jusqu'au complet pour garder la dernière transition
            for i in range(1, len(chords)+1):
                trie.insert(chords[:i])
    return trie

# (optionnel) si tu veux un bigram dict en plus
def build_bigrams(entries):
    from collections import Counter, defaultdict
    bigrams = defaultdict(Counter)
    for e in entries:
        c = e.get("chords", [])
        for a,b in zip(c, c[1:]):
            bigrams[a][b] += 1
    return bigrams

def load_and_build(jsonl_path):
    entries = load_musicbench_jsonl(jsonl_path)
    return build_harmony_trie(entries)
