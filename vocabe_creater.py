import pandas as pd
from tokenizer import ChemTokenizer
df = pd.read_csv("Data_FE.csv")
tokenizer = ChemTokenizer()



a_tokens = tokenizer.known_A_symbols
b_tokens = tokenizer.known_B_symbols    
x_tokens = tokenizer.known_X_symbols

all_tokens = tokenizer.known_A_symbols + tokenizer.known_B_symbols + tokenizer.known_X_symbols

vocab = {element: encode for element, encode in zip(all_tokens, range(len(all_tokens)))}
import json

print("Vocab:",vocab)
with open("canonical_vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)



print("Len of A",len(a_tokens))
print("\n")
print("Len of B",len(b_tokens))
# print(x_tokens)
from pprint import pprint
pprint("Composite Vocab:")
