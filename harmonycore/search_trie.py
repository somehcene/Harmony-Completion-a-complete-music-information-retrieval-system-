# === PATCH search_trie.py ===
from .trie import HarmonyTrie

def autocomplete_chords(trie, prefix, topk=5):
    # renvoie [(next_chord, count), ...]
    return trie.top_next(prefix, k=topk)
