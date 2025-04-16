import pandas as pd
import json
from pprint import pprint
from tokenizer import ChemTokenizer

# ── Load data & tokenizer ────────────────────────────────────────────────────
df = pd.read_csv("Data/Full_Data_ranamed_columns.csv")
tokenizer = ChemTokenizer()

a_tokens = tokenizer.known_A_symbols
b_tokens = tokenizer.known_B_symbols
x_tokens = tokenizer.known_X_symbols
all_tokens = a_tokens + b_tokens + x_tokens

# ── Build vocabulary ────────────────────────────────────────────────────────
vocab = {"None": 0}                                 # reserve 0
vocab.update({element: idx
              for idx, element in enumerate(all_tokens, start=1)})

# ── Persist to disk ──────────────────────────────────────────────────────────
with open("canonical_vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

# ── Quick sanity checks ──────────────────────────────────────────────────────
print("Vocab (truncated):")
pprint(dict(list(vocab.items())[:10]))              # show first 10 entries

print(f"\nLen of A: {len(a_tokens)}")
print(f"Len of B: {len(b_tokens)}")
print(f"Total vocab size (incl. 'None'): {len(vocab)}")
