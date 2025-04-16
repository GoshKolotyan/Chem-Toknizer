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
        self.logger = logging.getLogger("ChemTokenizer")
        self.logger.info("Initializing ChemTokenizer")
        
        # Load vocabulary
        try:
            with open('./Data/canonical_vocab.json', 'r') as f:
                self.vocab = json.load(f)
            self.logger.info(f"Vocabulary loaded with {len(self.vocab)} entries")
        except Exception as e:
            self.logger.error(f"Failed to load vocabulary: {e}")
            raise
        
        # Load element data
        try:
            self.known_A_elements = pd.read_csv('./Data/A_part_AA.csv', delimiter=';')
            self.known_B_elements = pd.read_csv('./Data/B_part_AA.csv', delimiter=';')
            self.known_X_elements = pd.read_csv('./Data/X_part_AA.csv', delimiter=';')
            self.logger.info(f"Element data loaded: A={len(self.known_A_elements)}, B={len(self.known_B_elements)}, X={len(self.known_X_elements)} elements")
        except Exception as e:
            self.logger.error(f"Failed to load element data: {e}")
            raise

        # Extract symbols and charges
        self.known_A_symbols = list(self.known_A_elements['Element'])
        self.known_B_symbols = list(self.known_B_elements['Element'])
        self.known_X_symbols = list(self.known_X_elements['Element'])

        self.known_A_charges = dict(zip(self.known_A_elements["Element"], self.known_A_elements["Charge"]))
        self.known_B_charges = dict(zip(self.known_B_elements['Element'], self.known_B_elements["Charge"]))
        self.known_X_charges = dict(zip(self.known_X_elements['Element'], self.known_X_elements["Charge"]))

        # Create unified charge dictionary and site mapping for efficiency
        self.element_charges = {**self.known_A_charges, **self.known_B_charges, **self.known_X_charges}
        self.element_sites = {}
        for element in self.known_A_symbols:
            self.element_sites[element] = "A-site"
        for element in self.known_B_symbols:
            self.element_sites[element] = "B-site"
        for element in self.known_X_symbols:
            self.element_sites[element] = "X-site"

        # Create regex pattern for tokenization
        multi_letter_symbols = self.known_A_symbols + self.known_B_symbols + self.known_X_symbols
        sorted_multi_symbols = sorted(multi_letter_symbols, key=len, reverse=True)
        multi_letter_pattern = "|".join(sorted_multi_symbols)
        
        self.pattern = re.compile(
            rf'(?:{multi_letter_pattern})|[A-Z][a-z]?|\d+(?:\.\d+)?'
        )
        self.logger.info("ChemTokenizer initialization completed")

    def encode(self, row):
        """Main encoding method that orchestrates the tokenization process."""
        self.logger.info(f"Encoding formula: {row.get('Name', 'unknown')}")
        
        try:
            # Extract data
            row_data = self._extract_row_data(row)
            
            # Get formula from name
            formula_str = self._get_formula_string(row_data["name_value"])
            
            # Tokenize formula
            raw_tokens = self._tokenize_formula(formula_str)
            self.logger.info(f"Tokenized '{formula_str}' into {len(raw_tokens)} tokens")
            
            # Process tokens
            element_probs = self._pair_elements_with_probs(raw_tokens)
            
            # Categorize tokens
            site_tokens = self._categorize_elements(element_probs)
            
            # Build sequence
            final_sequence = self._build_sequence(site_tokens, row_data)
            
            # Calculate charge balance
            charge_info = self._calculate_charge_balance(site_tokens)
            if not charge_info["is_balanced"]:
                self.logger.warning(f"Formula is not charge-balanced: total charge = {charge_info['total_charge']}")
                return None  # Return nothing if not charge balanced
            
            # Convert to tensor
            result = self._convert_to_tensor(final_sequence, row_data["bandgap"])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during encoding: {e}", exc_info=True)
            raise
    
    def _extract_row_data(self, row):
        """Extract data from the input row and handle missing values."""
        try:
            # Get required fields with defaults for missing values
            row_data = {
                "name_value": row["Name"],
                "r_a": row["R_a"],
                "r_b": row["R_b"],
                "r_x": row["R_x"],
                "prop_a_1": row.get("Prop_a_1", 0),
                "prop_a_2": row.get("Prop_a_2", 0),
                "prop_b_1": row.get("Prop_b_1", 0),
                "prop_b_2": row.get("Prop_b_2", 0),
                "prop_b_3": row.get("Prop_b_3", 0),
                "prop_x_1": row.get("Prop_x_1", 0),
                "prop_x_2": row.get("Prop_x_2", 0),
                "tf": row["Tolerance_Factor"],
                "structure": row["Structure_of_Material"],
                "bandgap": row["BandGap"]
            }
            
            # Convert None values to 0
            for key, value in row_data.items():
                if value is None and key != "name_value":
                    row_data[key] = 0
                    
        except KeyError as e:
            self.logger.error(f"Missing required column in data: {e}")
            raise
            
        return row_data
    
    def _get_formula_string(self, name_value):
        """Extract the formula string from the name field."""
        if isinstance(name_value, list) and len(name_value) > 0:
            formula_str = name_value[0]
        elif isinstance(name_value, str):
            formula_str = name_value
        else:
            error_msg = f"Invalid name format: {type(name_value)}"
            self.logger.warning(error_msg)
            raise ValueError(error_msg)
            
        return formula_str
    
    def _tokenize_formula(self, formula_str):
        """Tokenize the chemical formula string."""
        raw_tokens = self.pattern.findall(formula_str)
        return raw_tokens
    
    def _pair_elements_with_probs(self, raw_tokens):
        """Pair elements with their probabilities from the tokenized formula."""
        tokens = []
        i = 0
        while i < len(raw_tokens):
            element = raw_tokens[i]
            prob = 1.0  # default probability if not specified

            # Check if the next token is a number (including decimals)
            if i + 1 < len(raw_tokens) and re.fullmatch(r"\d+(?:\.\d+)?", raw_tokens[i + 1]):
                prob = float(raw_tokens[i + 1])
                i += 1  # skip the number on the next loop

            tokens.append((element, prob))
            i += 1

        return tokens
    
    def _categorize_elements(self, element_probs):
        """Categorize elements into A, B, X sites based on the known element lists."""
        site_tokens = {
            "a_tokens": {"Element": [], "Probs": []},
            "b_tokens": {"Element": [], "Probs": []},
            "x_tokens": {"Element": [], "Probs": []}
        }

        for element, prob in element_probs:
            if element in self.known_A_symbols:
                site_tokens["a_tokens"]["Element"].append(element)
                site_tokens["a_tokens"]["Probs"].append(prob)
            elif element in self.known_B_symbols:
                site_tokens["b_tokens"]["Element"].append(element)
                site_tokens["b_tokens"]["Probs"].append(prob)
            elif element in self.known_X_symbols:
                site_tokens["x_tokens"]["Element"].append(element)
                site_tokens["x_tokens"]["Probs"].append(prob)
            else:
                self.logger.warning(f"Unknown element: {element} - not found in A, B, or X lists")

        self.logger.debug(f"Site categorization: A={len(site_tokens['a_tokens']['Element'])}, "
                         f"B={len(site_tokens['b_tokens']['Element'])}, "
                         f"X={len(site_tokens['x_tokens']['Element'])} elements")
        
        return site_tokens
    
    def _build_sequence(self, site_tokens, row_data):
        """Build the sequence from categorized tokens and additional properties."""
        final_sequence = []
        tok = None

        # Process A-site tokens
        a_tokens = site_tokens["a_tokens"]
        if a_tokens['Element']:
            a_token = [self.vocab.get(token, tok) for token in a_tokens['Element']]

            if len(a_tokens["Element"]) == 1:
                a_token.append(0)
                a_part = [[a_token, row_data['prop_a_1'], row_data['prop_a_2'], row_data['r_a']]]
                final_sequence.extend(a_part)
            elif len(a_tokens["Element"]) == 2:
                a_part = [[a_token, row_data['prop_a_1'], row_data['prop_a_2'], row_data['r_a']]]
                final_sequence.extend(a_part)

        # Process B-site tokens
        b_tokens = site_tokens["b_tokens"]
        if b_tokens['Element']:
            b_token = [self.vocab.get(token, tok) for token in b_tokens["Element"]]

            if len(b_tokens["Element"]) == 1:
                b_token.append(0)
                b_part = [[b_token, row_data['prop_b_1'], row_data['prop_b_2'], row_data['prop_b_3'], row_data['r_b']]]
                final_sequence.extend(b_part)
            elif len(b_tokens["Element"]) == 2:
                b_part = [[b_token, row_data['prop_b_1'], row_data['prop_b_2'], row_data['prop_b_3'], row_data['r_b']]]
                final_sequence.extend(b_part)
            elif len(b_tokens["Element"]) == 3:
                self.logger.warning("Three B-site elements is unexpected")

        # Process X-site tokens
        x_tokens = site_tokens["x_tokens"]
        if x_tokens['Element']:
            x_token = [self.vocab.get(token, tok) for token in x_tokens["Element"]]
            
            if len(x_tokens["Element"]) == 1:
                x_token.append(0)
                x_part = [[x_token, row_data['prop_x_1'], row_data['prop_x_2'], row_data['r_x']]]
                final_sequence.extend(x_part)
            elif len(x_tokens["Element"]) == 2:    
                x_part = [[x_token, row_data['prop_x_1'], row_data['prop_x_2'], row_data['r_x']]]
                final_sequence.extend(x_part)
        
        # Add structure and tolerance factor
        final_sequence.append(row_data["structure"])
        final_sequence.append(row_data["tf"])
        
        return final_sequence
    
    def _calculate_charge_balance(self, site_tokens):
        """Calculate charge balance for the chemical formula."""
        # Combine all elements and probabilities
        all_elements = (site_tokens["a_tokens"]["Element"] + 
                         site_tokens["b_tokens"]["Element"] + 
                         site_tokens["x_tokens"]["Element"])
        all_probs = (site_tokens["a_tokens"]["Probs"] + 
                      site_tokens["b_tokens"]["Probs"] + 
                      site_tokens["x_tokens"]["Probs"])
        
        total_charge = 0
        charge_contributions = []
        
        for element, prob in zip(all_elements, all_probs):
            if element in self.element_charges:
                element_charge = self.element_charges[element]
                site = self.element_sites[element]
                contribution = prob * element_charge
                total_charge += contribution
                charge_contributions.append((element, prob, element_charge, contribution, site))
            else:
                self.logger.warning(f"No charge found for element {element}")
        
        charge_balanced = abs(total_charge) <= 0.01  # Allow small floating point errors
        
        return {
            "total_charge": total_charge,
            "is_balanced": charge_balanced,
            "contributions": charge_contributions
        }
    
    def _convert_to_tensor(self, final_sequence, bandgap):
        """Convert the final sequence to tensor representation."""
        try:
            # Flatten the nested sequence
            flat_seq = self._flatten_nested(final_sequence)
            
            # Convert to tensors
            torch_tensor = torch.FloatTensor(flat_seq)
            target = torch.tensor(bandgap)
            
            self.logger.debug(f"Created tensor of shape {torch_tensor.shape}")
            
            return torch_tensor, target
            
        except Exception as e:
            self.logger.error(f"Error creating tensor: {e}")
            raise
    
    def _flatten_nested(self, seq):
        """Flatten a nested list structure to a single list of floats."""
        result = []
        for item in seq:
            if isinstance(item, list):
                result.extend(self._flatten_nested(item))
            else:
                try:
                    result.append(float(item) if item is not None else 0.0)
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Failed to convert {item} to float: {e}")
                    result.append(0.0)  # Default value on error
        return result

    def decode(self, tokens):
        """
        Decode the tensor representation back to a chemical formula (to be implemented).
        """
        self.logger.info("Decode method called but not implemented")
        raise NotImplementedError("Decode method is not implemented yet")

# Example usage
if __name__ == "__main__":
    logger = logging.getLogger("ChemTokenizer_Main")
    logger.info("Starting ChemTokenizer example")
    
    try:
        logger.info("Loading dataset")
        df = pd.read_csv("Data/Full_Data_ranamed_columns.csv")
        logger.info(f"Loaded data with {len(df)} rows")
        
        tokenizer = ChemTokenizer()
        
        valid_count = 0
        invalid_count = 0
        
        # Process example rows
        for row_index in range(100):
            example = df.iloc[row_index]
            
            logger.info(f"Processing formula: {example['Name']}")
            result = tokenizer.encode(example)
            
            if result is None:
                logger.warning(f"Skipped formula {example['Name']} due to charge imbalance")
                invalid_count += 1
            else:
                X_, target = result
                logger.info(f"Result tensor shape: {X_.shape}, Target value: {target}")
                valid_count += 1
            
        logger.info(f"Processing complete. Valid formulas: {valid_count}, Invalid formulas: {invalid_count}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)