import re
import json
import logging
import pandas as pd
import torch
from time import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tokenizer.log"),
        logging.StreamHandler()
    ]
)

class ChemTokenizer:
    def __init__(self):
        start_time = time()
        self.logger = logging.getLogger("ChemTokenizer")
        self.logger.info("Initializing ChemTokenizer")
        
        # Load vocabulary
        try:
            with open('./Data/canonical_vocab.json', 'r') as f:
                self.vocab = json.load(f)
            self.logger.debug(f"Vocabulary loaded successfully with {len(self.vocab)} entries")
        except Exception as e:
            self.logger.error(f"Failed to load vocabulary: {e}")
            raise
        
        # Load element data
        try:
            self.known_A_elements = pd.read_csv('./Data/A_part_AA.csv', delimiter=';')
            self.known_B_elements = pd.read_csv('./Data/B_part_AA.csv', delimiter=';')
            self.known_X_elements = pd.read_csv('./Data/X_part_AA.csv', delimiter=';')
            self.logger.debug(f"Element data loaded successfully: A={len(self.known_A_elements)}, B={len(self.known_B_elements)}, X={len(self.known_X_elements)} elements")
        except Exception as e:
            self.logger.error(f"Failed to load element data: {e}")
            raise

        # Extract symbols and charges
        self.known_A_symbols = list(self.known_A_elements['Element'])
        self.known_B_symbols = list(self.known_B_elements['Element'])
        self.known_X_symbols = list(self.known_X_elements['Element'])
        self.logger.debug(f"Extracted element symbols: A={len(self.known_A_symbols)}, B={len(self.known_B_symbols)}, X={len(self.known_X_symbols)}")

        self.known_A_charges = dict(zip(self.known_A_elements["Element"], self.known_A_elements["Charge"]))
        self.known_B_charges = dict(zip(self.known_B_elements['Element'], self.known_B_elements["Charge"]))
        self.known_X_charges = dict(zip(self.known_X_elements['Element'], self.known_X_elements["Charge"]))
        self.logger.debug("Element charge dictionaries created")

        # Create regex pattern for tokenization
        multi_letter_symbols = self.known_A_symbols + self.known_B_symbols + self.known_X_symbols
        sorted_multi_symbols = sorted(multi_letter_symbols, key=len, reverse=True)
        multi_letter_pattern = "|".join(sorted_multi_symbols)
        
        self.pattern = re.compile(
            rf'(?:{multi_letter_pattern})|[A-Z][a-z]?|\d+(?:\.\d+)?'
        )
        self.logger.debug("Regex pattern compiled successfully")
        
        end_time = time()
        self.logger.info(f"ChemTokenizer initialization completed in {end_time - start_time:.4f} seconds")

    def encode(self, row):
        """
        Encodes chemical formula information into tensor representation.
        """
        encode_start_time = time()
        self.logger.info(f"Starting encoding for formula: {row.get('Name', 'unknown')}")
        
        # Extract data from row
        try:
            name_value = row["Name"]
            r_a = row["R_a"]
            r_b = row["R_b"]
            r_x = row["R_x"]
            prop_a_1 = row["Prop_a_1"] if row["Prop_a_1"] else 0
            prop_a_2 = row["Prop_a_2"] if row["Prop_a_2"] else 0
            prop_b_1 = row["Prop_b_1"] if row["Prop_b_1"] else 0 
            prop_b_2 = row["Prop_b_2"] if row["Prop_b_2"] else 0
            prop_b_3 = row["Prop_b_3"] if row["Prop_b_3"] else 0
            prop_x_1 = row["Prop_x_1"] if row["Prop_x_1"] else 0
            prop_x_2 = row["Prop_x_2"] if row["Prop_x_2"] else 0
            tf = row["Tolerance_Factor"]
            structure = row["Structure_of_Material"]
            bandgap = row["BandGap"]
            self.logger.debug(f"Extracted row data: r_a={r_a}, r_b={r_b}, r_x={r_x}, structure={structure}, tf={tf}, bandgap={bandgap}")
        except KeyError as e:
            self.logger.error(f"Missing required column in data: {e}")
            raise

        # Step 1: Handle 'Name' if it's a list or a string
        step_start = time()
        if isinstance(name_value, list) and len(name_value) > 0:
            formula_str = name_value[0]
            self.logger.debug(f"Name is a list, using first element: {formula_str}")
        elif isinstance(name_value, str):
            formula_str = name_value
            self.logger.debug(f"Name is a string: {formula_str}")
        else:
            self.logger.warning(f"Invalid name format: {type(name_value)}, skipping")
            return []
        self.logger.debug(f"Step 1 completed in {time() - step_start:.4f} seconds")

        # Step 2: Tokenize the formula string
        step_start = time()
        raw_tokens = self.pattern.findall(formula_str)
        self.logger.info(f"Tokenized formula '{formula_str}' into {len(raw_tokens)} raw tokens: {raw_tokens}")
        self.logger.debug(f"Step 2 completed in {time() - step_start:.4f} seconds")
            
        # Step 3: Pair elements with their probabilities
        step_start = time()
        tokens = []
        i = 0
        while i < len(raw_tokens):
            element = raw_tokens[i]
            prob = 1.0  # default probability if not specified

            # Check if the next token is a number (including decimals)
            if i + 1 < len(raw_tokens) and re.fullmatch(r"\d+(?:\.\d+)?", raw_tokens[i + 1]):
                prob = float(raw_tokens[i + 1])
                self.logger.debug(f"Found probability {prob} for element {element}")
                i += 1  # skip the number on the next loop
            else:
                self.logger.debug(f"No probability specified for element {element}, using default {prob}")

            tokens.append((element, prob))
            i += 1

        self.logger.info(f"Paired elements with probabilities: {tokens}")
        self.logger.debug(f"Step 3 completed in {time() - step_start:.4f} seconds")

        # Step 4: Categorize elements into A, B, X sites
        step_start = time()
        a_tokens = {"Element":[], "Probs":[]}
        b_tokens = {"Element":[], "Probs":[]}
        x_tokens = {"Element":[], "Probs":[]}

        for element, prob in tokens:
            if element in self.known_A_symbols:
                a_tokens["Element"].append(element)
                a_tokens["Probs"].append(prob)
                self.logger.debug(f"Categorized {element} as A-site element with probability {prob}")
            elif element in self.known_B_symbols:
                b_tokens["Element"].append(element)
                b_tokens["Probs"].append(prob)
                self.logger.debug(f"Categorized {element} as B-site element with probability {prob}")
            elif element in self.known_X_symbols:
                x_tokens["Element"].append(element)
                x_tokens["Probs"].append(prob)
                self.logger.debug(f"Categorized {element} as X-site element with probability {prob}")
            else:
                self.logger.warning(f"Unknown element: {element} - not found in A, B, or X lists")

        self.logger.info(f"A-site elements: {a_tokens['Element']} with probs {a_tokens['Probs']}")
        self.logger.info(f"B-site elements: {b_tokens['Element']} with probs {b_tokens['Probs']}")
        self.logger.info(f"X-site elements: {x_tokens['Element']} with probs {x_tokens['Probs']}")
        self.logger.debug(f"Step 4 completed in {time() - step_start:.4f} seconds")

        # Step 5: Build the final sequence
        step_start = time()
        final_sequence = []
        tok = None

        # Process A-site tokens
        if a_tokens['Element']:
            a_token = [self.vocab.get(token, tok) for token in a_tokens['Element']]
            self.logger.debug(f"A tokens: {a_tokens['Element']}, Encoded as: {a_token}")

            if len(a_tokens["Element"]) == 1:
                self.logger.debug(f"Single A-site element, using prop_a_1={prop_a_1}, r_a={r_a}")
                a_part = [[a_token, prop_a_1, prop_a_2, r_a]]
                final_sequence.extend(a_part)
            elif len(a_tokens["Element"]) == 2:
                self.logger.debug(f"Two A-site elements, using prop_a_1={prop_a_1}, prop_a_2={prop_a_2}, r_a={r_a}")
                a_part = [[a_token, prop_a_1, prop_a_2, r_a]]
                final_sequence.extend(a_part)
            self.logger.info(f"A-site processing complete, sequence now: {final_sequence}")

        # Process B-site tokens
        if b_tokens['Element']:
            b_token = [self.vocab.get(token, tok) for token in b_tokens["Element"]]
            self.logger.debug(f"B tokens: {b_tokens['Element']}, Encoded as: {b_token}")

            if len(b_tokens["Element"]) == 1:
                self.logger.debug(f"Single B-site element, using prop_b_1={prop_b_1}, r_b={r_b}")
                b_part = [[b_token, prop_b_1, prop_b_2, prop_b_3, r_b]]
                final_sequence.extend(b_part)
            elif len(b_tokens["Element"]) == 2:
                self.logger.debug(f"Two B-site elements, using prop_b_1={prop_b_1}, prop_b_2={prop_b_2}, r_b={r_b}")
                b_part = [[b_token, prop_b_1, prop_b_2, prop_b_3, r_b]]
                final_sequence.extend(b_part)
            elif len(b_tokens["Element"]) == 3:
                self.logger.debug(f"Three B-site elements, using prop_b_1={prop_b_1}, prop_b_2={prop_b_2}, prop_b_3={prop_b_3}, r_b={r_b}")
                b_part = [[b_token, prop_b_1, prop_b_2, prop_b_3, r_b]]
                final_sequence.extend(b_part)
            self.logger.info(f"B-site processing complete, sequence now: {final_sequence}")

        # Process X-site tokens
        if x_tokens['Element']:
            x_token = [self.vocab.get(token, tok) for token in x_tokens["Element"]]
            self.logger.debug(f"X tokens: {x_tokens['Element']}, Encoded as: {x_token}")
            
            if len(x_tokens["Element"]) == 1:
                self.logger.debug(f"Single X-site element, using prop_x_1={prop_x_1}, r_x={r_x}")
                x_part = [[x_token, prop_x_1, prop_x_2, r_x]]
                final_sequence.extend(x_part)
            elif len(x_tokens["Element"]) == 2:    
                self.logger.debug(f"Two X-site elements, using prop_x_1={prop_x_1}, prop_x_2={prop_x_2}, r_x={r_x}")
                x_part = [[x_token, prop_x_1, prop_x_2, r_x]]
                final_sequence.extend(x_part)
            self.logger.info(f"X-site processing complete, sequence now: {final_sequence}")
        
        self.logger.debug(f"Step 5 completed in {time() - step_start:.4f} seconds")

        # Step 6: Calculate charge balance
        step_start = time()
        finall_elements_list = a_tokens['Element'] + b_tokens['Element'] + x_tokens['Element']
        finall_probs_list = a_tokens['Probs'] + b_tokens['Probs'] + x_tokens['Probs']
        self.logger.debug(f"Combined elements list: {finall_elements_list}")
        self.logger.debug(f"Combined probabilities list: {finall_probs_list}")
        
        charge = 0
        charge_contributions = []
        
        for index in range(len(finall_elements_list)):
            element = finall_elements_list[index]
            prob = finall_probs_list[index]
            element_charge = None
            source = ""
            
            if self.known_A_charges.get(element):
                element_charge = self.known_A_charges.get(element)
                source = "A-site"
            elif self.known_B_charges.get(element):
                element_charge = self.known_B_charges.get(element)
                source = "B-site"
            elif self.known_X_charges.get(element):
                element_charge = self.known_X_charges.get(element)
                source = "X-site"
                
            if element_charge is not None:
                contribution = prob * element_charge
                charge += contribution
                charge_contributions.append((element, prob, element_charge, contribution, source))
                self.logger.debug(f"Element: {element} ({source}), Prob: {prob}, Charge: {element_charge}, Contribution: {contribution}, Running total: {charge}")
            else:
                self.logger.warning(f"No charge found for element {element}")
        
        # Log summary of charge calculation
        self.logger.info("Charge balance calculation:")
        for element, prob, charge_val, contribution, source in charge_contributions:
            self.logger.info(f"  {element} ({source}): {prob} × {charge_val} = {contribution}")
        self.logger.info(f"Total charge: {charge}")
        
        if abs(charge) > 0.01:  # Allow small floating point errors
            self.logger.warning(f"Formula may not be charge-balanced: total charge = {charge}")
        else:
            self.logger.info("Formula appears to be charge-balanced")
        
        self.logger.debug(f"Step 6 completed in {time() - step_start:.4f} seconds")

        # Step 7: Add remaining features and convert to tensor
        step_start = time()
        final_sequence.append(structure)
        final_sequence.append(tf)
        # final_sequence.append(bandgap)

        self.logger.debug(f"Added structure ({structure}), tolerance factor ({tf}), and bandgap ({bandgap}) to sequence")
        
        # Convert to tensor
        try:
            def flatten_nested(seq):
                result = []
                for item in seq:
                    if isinstance(item, list):
                        result.extend(flatten_nested(item))
                    else:
                        try:
                            result.append(float(item))
                        except (ValueError, TypeError) as e:
                            self.logger.error(f"Failed to convert {item} to float: {e}")
                            result.append(0.0)  # Default value on error
                return result

            flat_seq = flatten_nested(final_sequence)
            self.logger.debug(f"Flattened sequence length: {len(flat_seq)}")
            self.logger.debug(f"Flattened values: {flat_seq}")
            
            torch_tensor = torch.FloatTensor(flat_seq)
            target = torch.tensor(bandgap)
            
            self.logger.info(f"Created tensor of shape {torch_tensor.shape}")
            self.logger.info(f"Target value (bandgap): {target}")
        except Exception as e:
            self.logger.error(f"Error creating tensor: {e}")
            raise
        
        self.logger.debug(f"Step 7 completed in {time() - step_start:.4f} seconds")
        
        encode_end_time = time()
        self.logger.info(f"Encoding completed in {encode_end_time - encode_start_time:.4f} seconds")
        return torch_tensor, target

    def decode(self, tokens):
        """
        Decode the tensor representation back to a chemical formula (if implemented).
        """
        self.logger.info("Decode method called but not implemented")
        pass

# Example usage
if __name__ == "__main__":
    start_time = time()
    logger = logging.getLogger("ChemTokenizer_Main")
    logger.info("Starting ChemTokenizer example")
    
    try:
        logger.info("Loading dataset")
        df = pd.read_csv("Data/Full_Data_ranamed_columns.csv")
        logger.info(f"Loaded data with {len(df)} rows")
        
        logger.info("Initializing tokenizer")
        tokenizer = ChemTokenizer()
        
        # Process an example row
        for row_index in range(10):
            logger.info(f"Processing example row {row_index}")
            example = df.iloc[row_index]
            
            logger.info(f"Processing formula: {example['Name']}")
            X_, target = tokenizer.encode(example)

            logger.info(f"Result tensot X part {X_}")        
            logger.info(f"Result tensor shape: {X_.shape}")
            logger.info(f"Target value (bandgap): {target}")
            
            end_time = time()
            logger.info(f"Total execution time: {end_time - start_time:.4f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)