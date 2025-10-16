# === PATCH trie.py ===
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end = False
        self.count = 0

class HarmonyTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, chord_seq):
        node = self.root
        for chord in chord_seq:
            node = node.children[chord]
            node.count += 1
        node.is_end = True

    def _descend(self, prefix):
        node = self.root
        for chord in prefix:
            if chord not in node.children:
                return None
            node = node.children[chord]
        return node

    # >>> NOUVEAU : compter les enfants immédiats (prochains accords) avec leurs fréquences
    def next_counts(self, prefix):
        node = self._descend(prefix)
        if node is None:
            return {}
        return {ch: child.count for ch, child in node.children.items()}

    # >>> NOUVEAU : top-N prochains accords
    def top_next(self, prefix, k=5):
        counts = self.next_counts(prefix)
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]

    # (ta méthode autocomplete actuelle reste telle quelle)
    def autocomplete(self, prefix):
        node = self._descend(prefix)
        if node is None:
            return []
        return self._collect(node, prefix)

    def _collect(self, node, prefix):
        results = []
        if node.is_end:
            results.append((prefix, node.count))
        for chord, child in node.children.items():
            results.extend(self._collect(child, prefix + [chord]))
        return results
