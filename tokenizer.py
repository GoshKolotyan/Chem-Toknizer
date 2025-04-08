import re
import pandas as pd
import torch

class ChemTokenizer:
    def __init__(self):
        self.special_tokens = ["CLS", "SEP"]

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
        prob_a  = row["Prop_a"]    # or "Prob_a" if that's your actual CSV column
        prob_b  = row["Prop_b"]
        prob_b2 = row["Prop_b_2"]  # or "Prov_b_2" if that matches your CSV
        prob_x  = row["Prop_x"]
        tf      = row["Tolerance_Factor"]
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

        # -- Separate tokens by site: A, B, or X. This helps us group them under "A_Part", etc. --
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
                # If not found, you can decide how to handle it. We'll ignore or store in a separate list:
                # others.append(t)
                pass

        final_sequence = []

        # --- A-site tokens block ---
        if a_tokens:
            final_sequence.append("A_Part")
            for t in a_tokens:
                # e.g. "MA", then [prob_a], [r_a]
                final_sequence.append(t)           # "MA"
                final_sequence.append([prob_a])    # [1]
                final_sequence.append([r_a])       # [2.16]

        # --- B-site tokens block ---
        if b_tokens:
            final_sequence.append("B_Part")
            for t in b_tokens:
                final_sequence.append(t)           # "Pb"
                final_sequence.append([prob_b])    # [1]
                final_sequence.append([prob_b2])   # [0]
                final_sequence.append([r_b])       # [1.19]

        # --- X-site tokens block ---
        if x_tokens:
            final_sequence.append("X_part")
            for t in x_tokens:
                final_sequence.append(t)          # "I"
                final_sequence.append([prob_x])   # [1]
                final_sequence.append([r_x])      # [2.2]

        # Finally, append structure, tf, and bandgap as single-element lists
        # Example: [0], [0.909435270198628], [1.55]
        final_sequence.append([structure])
        final_sequence.append([tf])
        final_sequence.append([bandgap])

        return final_sequence

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
    df = pd.read_csv("Data_FE.csv")

    tokenizer = ChemTokenizer()
    example = df.iloc[0]  # first row as an example

    result = tokenizer.encode(example)
    print(result)

    # row = df.iloc[-1].to_dict()  # first row as an example
    example = {
        "Name": ["MA0.1FA0.9PbI3"],
        "R_a": [2.16],
        "Prop_a": [1],
        "R_b": [1.19],
        "Prop_b": [1],
        "Prop_b_2": [0],
        "R_x": [2.2],
        "Prop_x": [1],
        "BandGap": [1.55],
        "Tolerance_Factor": [0.909435270198628],
        "Structure_of_Material": [0],
    }

    print(tokenizer.encode(example))
