import re
import json
from pprint import pprint
import pandas as pd
import torch

class ChemTokenizer:
    def __init__(self):
        # self.special_tokens = ["CLS", "SEP"]
        with open('canonical_vocab.json', 'r') as f:
            self.vocab = json.load(f)
        # print("Vocab:", self.vocab) 
        # 1) Read CSV data for A/B/X
        self.known_A_elements = pd.read_csv('A_part_AA.csv', delimiter=';')
        self.known_B_elements = pd.read_csv('B_part_AA.csv', delimiter=';')
        self.known_X_elements = pd.read_csv('X_part_AA.csv', delimiter=';')

        # 2) Extract the 'Element' columns
        self.known_A_symbols = list(self.known_A_elements['Element'])
        self.known_B_symbols = list(self.known_B_elements['Element'])
        self.known_X_symbols = list(self.known_X_elements['Element'])

        # 3) Create a combined list of multi-letter symbols for regex
        multi_letter_symbols = self.known_A_symbols + self.known_B_symbols + self.known_X_symbols
        # Sort descending by length so multi-letter tokens (like "MA") match first
        sorted_multi_symbols = sorted(multi_letter_symbols, key=len, reverse=True)

        # 4) Build the alternation pattern
        multi_letter_pattern = "|".join(sorted_multi_symbols)

        # 5) Compile the pattern: match known multi-letter tokens OR standard elements OR digits
        #    If you need to handle decimal (e.g. 0.1) coefficients, add \d+\.\d+ as well.
        self.pattern = re.compile(
            rf'(?:{multi_letter_pattern})|[A-Z][a-z]?|\d+'
        )

    def encode(self, row):
        """
        Produces a list like:
          [
            "A_Part",
               <A token>, [prob_a], [r_a], 
               <A token>, [prob_a], [r_a],
            "B_Part",
               <B token>, [prob_b], [prob_b_2], [r_b],
            "X_part",
               <X token>, [prob_x], [r_x],
            [structure], [tf], [bandgap]
          ]
        Example:
          ['A_Part', 'MA', [1], [2.16], 'FA', [1], [2.16], 'B_Part', 'Pb', [1], [0], [1.19], 
           'X_part', 'I', [1], [2.2], [0], [0.909435270198628], [1.55]]
        """

        # -- Extract data from row --
        name_value = row["Name"]
        # Make sure these column names match your CSV
        r_a     = row["R_a"]
        r_b     = row["R_b"]
        r_x     = row["R_x"]
        prop_a_1  = row["Prop_a_1"]    # or "Prob_a" if that's your actual CSV column
        prop_a_2  = row["Prop_a_2"]    # or "Prob_a_2" if that's your actual CSV column
        prop_b_1  = row["Prop_b_1"]
        prop_b_2  = row["Prop_b_2"] 
        prop_b_3  = row["Prop_b_3"]  # or "Prov_b_2" if that matches your CSV
        prop_x_1  = row["Prop_x_1"]
        prop_x_2  = row["Prop_x_2"]
        tf        = row["Tolerance_Factor"]
        structure = row["Structure_of_Material"]
        bandgap   = row["BandGap"]

        # Handle 'Name' if it's a list or a string
        if isinstance(name_value, list) and len(name_value) > 0:
            formula_str = name_value[0]
        elif isinstance(name_value, str):
            formula_str = name_value
        else:
            return []  # or handle error

        # -- Tokenize the formula string (e.g., "MA0.1FA0.9PbI3" -> ["MA", "FA", "Pb", "I", "3"]) --
        raw_tokens = self.pattern.findall(formula_str)

        # -- Filter out digits (like '3'); you could also replicate tokens if desired --
        tokens = []
        for tok in raw_tokens:
            if tok.isdigit():
                continue
            tokens.append(tok)
        a_tokens = []
        b_tokens = []
        x_tokens = []

        for t in tokens:
            if t in self.known_A_symbols:
                a_tokens.append(t)
            elif t in self.known_B_symbols:
                b_tokens.append(t)
            elif t in self.known_X_symbols:
                x_tokens.append(t)
            else:

                pass

        final_sequence = []
        print("A_tokens:", a_tokens)
        print("B_tokens:", b_tokens)
        print("X_tokens:", x_tokens)
        # final_sequence.append("Composite")
        # --- A-site tokens block ---
        if a_tokens:
            # final_sequence.append("A_Part")
            if len(a_tokens) == 1:
                a_token = [self.vocab.get(token, tok) for token in a_tokens]
                print("A token is ",a_tokens, "Encode is", a_token)
    
                a_part = [[a_token, prop_a_1, r_a] for t in a_tokens]
                final_sequence.extend(a_part)
            elif len(a_tokens) ==2:
                a_token = [self.vocab.get(token, tok) for token in a_tokens]
                print("A token is ",a_tokens, "Encode is", a_token)
                a_part = [[a_token, prop_a_1, prop_a_2, r_a]]
                final_sequence.extend(a_part)
        # --- B-site tokens block ---
        if b_tokens:
            # final_sequence.append("B_Part")
            if len(b_tokens) == 1:
                b_token = [self.vocab.get(token, tok) for token in b_tokens]
                print("B token is ",b_tokens, "Encode is", b_token) 
                b_part = [[b_token, prop_b_1, r_b]]
                final_sequence.extend(b_part)
            elif len(b_tokens) == 2:
                b_token = [self.vocab.get(token, tok) for token in b_tokens]
                print("B token is ",b_tokens, "Encode is", b_token) 
                b_part = [[b_token, prop_b_1, prop_b_2, r_b]]
                final_sequence.extend(b_part)
            elif len(b_tokens) == 3:
                b_token = [self.vocab.get(token, tok) for token in b_tokens]
                print("B token is ",b_tokens, "Encode is", b_token) 
                b_part = [[b_token, prop_b_1,prop_b_2,prop_b_3, r_b]]
                final_sequence.extend(b_part)

        # # --- X-site tokens block ---
        if x_tokens:
            # final_sequence.append("X_part")
            x_token = [self.vocab.get(token, tok) for token in x_tokens]
            if len(x_tokens) == 1:
                x_part = [[x_token, prop_x_1, r_x]]
                print("X token is ",x_tokens, "Encode is", x_token) 
                final_sequence.extend(x_part)
            elif len(x_tokens) == 2:    
                print("X token is ",x_tokens, "Encode is", x_token) 
                x_part = [[x_token, prop_x_1, prop_x_2, r_x]]
                final_sequence.extend(x_part)

        # Finally, append structure, tf, and bandgap as single-element lists
        # Example: [0], [0.909435270198628], [1.55]
        final_sequence.append(structure)
        final_sequence.append(tf)
        # final_sequence.append("Target")
        final_sequence.append(bandgap)
        print(final_sequence)  # Debug: show the list you're converting
        def flatten_nested(seq):
            result = []
            for item in seq:
                if isinstance(item, list):
                    result.extend(flatten_nested(item))
                else:
                    result.append(float(item))
            return result

        flat_seq = flatten_nested(final_sequence)
        torch_tensor = torch.FloatTensor(flat_seq)
        target = torch_tensor[-1]
        return torch_tensor, target

    def decode(self, tokens):
        # Implement if you need a reverse operation
        pass

    def __len__(self):
        # Implement if you need a dataset length
        pass

    def __getitem__(self, idx):
        # Implement if you want to iterate or index into a dataset
        pass

# --- Example usage ---
if __name__ == "__main__":
#     # df = pd.read_csv("Data_FE.csv")

    tokenizer = ChemTokenizer()
    # example = df.iloc[0]  # first row as an example

    # example = df.iloc[-1].to_dict()  # first row as an example
    example = {
        "Name": ["Ba3CsGa5Se10Cl2"],#"Cs2AgBi0.25In0.5Sb0.25Br6"],
        "R_a": [2.16],
        "Prop_a_1": [1],
        "Prop_a_2": [0],
        "R_b": [1.19],
        "Prop_b_1": [1],
        "Prop_b_2": [0],
        "Prop_b_3": [0],
        "R_x": [2.2],
        "Prop_x_1": [1],
        "Prop_x_2": [0],
        "BandGap": [1.55],
        "Tolerance_Factor": [0.909435270198628],
        "Structure_of_Material": [0],
    }

    X_, traget = tokenizer.encode(example)
    print("X is --->",X_)
    print("BandGap is--->",traget)
