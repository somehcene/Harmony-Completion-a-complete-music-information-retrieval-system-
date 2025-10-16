import os
import pickle
from .build_trie import load_and_build

def main():
    CURRENT_DIR = os.path.dirname(__file__)
    jsonl_path = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Data", "MusicBench_train.json"))

    trie = load_and_build(jsonl_path)
    if trie:
        print("Trie built successfully.")

        # on sauvegarde le trie
        output_path = os.path.join(CURRENT_DIR, "..", "outputs", "harmony_trie.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  
        with open(output_path, "wb") as f:
            pickle.dump(trie, f)
            print(f"Trie saved to {output_path}")
    else:
        print("Failed to build trie.")

if __name__ == "__main__":
    main()

