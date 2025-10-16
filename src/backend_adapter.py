# src/backend_adapter.py
import pickle
from pathlib import Path
from typing import List, Tuple
from .notation import variants_for_chord

def softmax(xs, t=1.0):
    import math
    if not xs:
        return []
    m = max(xs)
    ex = [math.exp((x - m)/t) for x in xs]
    s = sum(ex) or 1.0
    return [e/s for e in ex]

class TrieSuggest:
    def __init__(self, trie):
        self.trie = trie

    @classmethod
    def from_pkl(cls, pkl_path: str|Path):
        with open(pkl_path, "rb") as f:
            trie = pickle.load(f)
        return cls(trie)

    def _top_next_raw(self, prefix: List[str], k=5):
        if hasattr(self.trie, "top_next"):
            return self.trie.top_next(prefix, k=k) or []
        return []

    def top_next(self, prefix: List[str], k=5, debug: bool=False) -> List[Tuple[str,int]]:
        """Backoff + variantes (enharmoniques / slash) pour éviter les trous."""
        def try_with(pref):
            res = self._top_next_raw(pref, k=k)
            if debug:
                print(f"[trie] try {pref} -> {res[:3]}")
            return res

        # 1) tel quel
        res = try_with(prefix)
        if res:
            return res

        # 2) variantes du dernier accord
        if prefix:
            head, last = prefix[:-1], prefix[-1]
            for v in variants_for_chord(last):
                if v == last:
                    continue
                res = try_with(head + [v])
                if res:
                    return res

        # 3) backoff (n-1, n-2, ...) + variantes
        for cut in range(len(prefix)-1, 0, -1):
            sub = prefix[-cut:]
            res = try_with(sub)
            if res:
                return res
            head, last = sub[:-1], sub[-1]
            for v in variants_for_chord(last):
                res = try_with(head + [v])
                if res:
                    return res

        return []

    def ensemble_rank(self, prefix: List[str], k=5, tonal_fn=None, alpha=0.7, debug=False):
        """
        Combine stats du trie (counts -> softmax) + score tonal (0..1).
        alpha = poids du trie ; (1-alpha) = poids tonal
        """
        base = self.top_next(prefix, k=max(k, 12), debug=debug)  # on prend un peu large
        if not base:
            return []  # laisser la CLI gérer le fallback
        labels, counts = zip(*base)
        p_trie = softmax([float(c) for c in counts], t=1.0)
        p_tonal = []
        prev = prefix[-1] if prefix else None
        for lab in labels:
            p_tonal.append(float(tonal_fn(prev, lab)) if tonal_fn else 0.0)
        scored = []
        for lab, a, b in zip(labels, p_trie, p_tonal):
            score = alpha * a + (1 - alpha) * b
            scored.append((lab, score, {"p_trie": round(a,4), "p_ton": round(b,4)}))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
