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

        # Create unified charge dictionary and site mapping for efficiency
        self.element_charges = {**self.known_A_charges, **self.known_B_charges, **self.known_X_charges}
        self.element_sites = {}
        for element in self.known_A_symbols:
            self.element_sites[element] = "A-site"
        for element in self.known_B_symbols:
            self.element_sites[element] = "B-site"
        for element in self.known_X_symbols:
            self.element_sites[element] = "X-site"
        self.logger.debug("Unified charge dictionary and site mapping created")

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
        """Main encoding method that orchestrates the tokenization process."""
        encode_start_time = time()
        self.logger.info(f"Starting encoding for formula: {row.get('Name', 'unknown')}")
        
        try:
            # Extract data
            row_data = self._extract_row_data(row)
            
            # Get formula from name
            formula_str = self._get_formula_string(row_data["name_value"])
            
            # Tokenize formula
            raw_tokens = self._tokenize_formula(formula_str)
            
            # Process tokens
            element_probs = self._pair_elements_with_probs(raw_tokens)
            
            # Categorize tokens
            site_tokens = self._categorize_elements(element_probs)
            
            # Build sequence
            final_sequence = self._build_sequence(site_tokens, row_data)
            
            # Calculate charge balance
            charge_info = self._calculate_charge_balance(site_tokens)
            
            # Convert to tensor
            result = self._convert_to_tensor(final_sequence, row_data["bandgap"])
            
            encode_end_time = time()
            self.logger.info(f"Encoding completed in {encode_end_time - encode_start_time:.4f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during encoding: {e}", exc_info=True)
            raise
    
    def _extract_row_data(self, row):
        """Extract data from the input row and handle missing values."""
        step_start = time()
        self.logger.debug("Extracting row data")
        
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
                    
            self.logger.debug(f"Extracted row data: r_a={row_data['r_a']}, r_b={row_data['r_b']}, r_x={row_data['r_x']}, structure={row_data['structure']}, tf={row_data['tf']}, bandgap={row_data['bandgap']}")
            
        except KeyError as e:
            self.logger.error(f"Missing required column in data: {e}")
            raise
            
        self.logger.debug(f"Step '_extract_row_data' completed in {time() - step_start:.4f} seconds")
        return row_data
    
    def _get_formula_string(self, name_value):
        """Extract the formula string from the name field."""
        step_start = time()
        self.logger.debug("Getting formula string from name value")
        
        if isinstance(name_value, list) and len(name_value) > 0:
            formula_str = name_value[0]
            self.logger.debug(f"Name is a list, using first element: {formula_str}")
        elif isinstance(name_value, str):
            formula_str = name_value
            self.logger.debug(f"Name is a string: {formula_str}")
        else:
            error_msg = f"Invalid name format: {type(name_value)}"
            self.logger.warning(error_msg)
            raise ValueError(error_msg)
            
        self.logger.debug(f"Step '_get_formula_string' completed in {time() - step_start:.4f} seconds")
        return formula_str
    
    def _tokenize_formula(self, formula_str):
        """Tokenize the chemical formula string."""
        step_start = time()
        self.logger.debug(f"Tokenizing formula: {formula_str}")
        
        raw_tokens = self.pattern.findall(formula_str)
        self.logger.info(f"Tokenized formula '{formula_str}' into {len(raw_tokens)} raw tokens: {raw_tokens}")
        
        self.logger.debug(f"Step '_tokenize_formula' completed in {time() - step_start:.4f} seconds")
        return raw_tokens
    
    def _pair_elements_with_probs(self, raw_tokens):
        """Pair elements with their probabilities from the tokenized formula."""
        step_start = time()
        self.logger.debug("Pairing elements with probabilities")
        
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
        self.logger.debug(f"Step '_pair_elements_with_probs' completed in {time() - step_start:.4f} seconds")
        return tokens
    
    def _categorize_elements(self, element_probs):
        """Categorize elements into A, B, X sites based on the known element lists."""
        step_start = time()
        self.logger.debug("Categorizing elements into A, B, X sites")
        
        site_tokens = {
            "a_tokens": {"Element": [], "Probs": []},
            "b_tokens": {"Element": [], "Probs": []},
            "x_tokens": {"Element": [], "Probs": []}
        }

        for element, prob in element_probs:
            if element in self.known_A_symbols:
                site_tokens["a_tokens"]["Element"].append(element)
                site_tokens["a_tokens"]["Probs"].append(prob)
                self.logger.debug(f"Categorized {element} as A-site element with probability {prob}")
            elif element in self.known_B_symbols:
                site_tokens["b_tokens"]["Element"].append(element)
                site_tokens["b_tokens"]["Probs"].append(prob)
                self.logger.debug(f"Categorized {element} as B-site element with probability {prob}")
            elif element in self.known_X_symbols:
                site_tokens["x_tokens"]["Element"].append(element)
                site_tokens["x_tokens"]["Probs"].append(prob)
                self.logger.debug(f"Categorized {element} as X-site element with probability {prob}")
            else:
                self.logger.warning(f"Unknown element: {element} - not found in A, B, or X lists")

        self.logger.info(f"A-site elements: {site_tokens['a_tokens']['Element']} with probs {site_tokens['a_tokens']['Probs']}")
        self.logger.info(f"B-site elements: {site_tokens['b_tokens']['Element']} with probs {site_tokens['b_tokens']['Probs']}")
        self.logger.info(f"X-site elements: {site_tokens['x_tokens']['Element']} with probs {site_tokens['x_tokens']['Probs']}")
        
        self.logger.debug(f"Step '_categorize_elements' completed in {time() - step_start:.4f} seconds")
        return site_tokens
    
    def _build_sequence(self, site_tokens, row_data):
        """Build the sequence from categorized tokens and additional properties."""
        step_start = time()
        self.logger.debug("Building the final sequence")
        
        final_sequence = []
        tok = None

        # Process A-site tokens
        a_tokens = site_tokens["a_tokens"]
        if a_tokens['Element']:
            a_token = [self.vocab.get(token, tok) for token in a_tokens['Element']]
            self.logger.debug(f"A tokens: {a_tokens['Element']}, Encoded as: {a_token}")

            if len(a_tokens["Element"]) == 1:
                self.logger.debug(f"Single A-site element, using prop_a_1={row_data['prop_a_1']}, r_a={row_data['r_a']}")
                a_part = [[a_token, row_data['prop_a_1'], row_data['prop_a_2'], row_data['r_a']]]
                final_sequence.extend(a_part)
            elif len(a_tokens["Element"]) == 2:
                self.logger.debug(f"Two A-site elements, using prop_a_1={row_data['prop_a_1']}, prop_a_2={row_data['prop_a_2']}, r_a={row_data['r_a']}")
                a_part = [[a_token, row_data['prop_a_1'], row_data['prop_a_2'], row_data['r_a']]]
                final_sequence.extend(a_part)
            self.logger.info(f"A-site processing complete, sequence now: {final_sequence}")

        # Process B-site tokens
        b_tokens = site_tokens["b_tokens"]
        if b_tokens['Element']:
            b_token = [self.vocab.get(token, tok) for token in b_tokens["Element"]]
            self.logger.debug(f"B tokens: {b_tokens['Element']}, Encoded as: {b_token}")

            if len(b_tokens["Element"]) == 1:
                self.logger.debug(f"Single B-site element, using prop_b_1={row_data['prop_b_1']}, r_b={row_data['r_b']}")
                b_part = [[b_token, row_data['prop_b_1'], row_data['prop_b_2'], row_data['prop_b_3'], row_data['r_b']]]
                final_sequence.extend(b_part)
            elif len(b_tokens["Element"]) == 2:
                self.logger.debug(f"Two B-site elements, using prop_b_1={row_data['prop_b_1']}, prop_b_2={row_data['prop_b_2']}, r_b={row_data['r_b']}")
                b_part = [[b_token, row_data['prop_b_1'], row_data['prop_b_2'], row_data['prop_b_3'], row_data['r_b']]]
                final_sequence.extend(b_part)
            elif len(b_tokens["Element"]) == 3:
                self.logger.debug(f"Three B-site elements, using prop_b_1={row_data['prop_b_1']}, prop_b_2={row_data['prop_b_2']}, prop_b_3={row_data['prop_b_3']}, r_b={row_data['r_b']}")
                b_part = [[b_token, row_data['prop_b_1'], row_data['prop_b_2'], row_data['prop_b_3'], row_data['r_b']]]
                final_sequence.extend(b_part)
            self.logger.info(f"B-site processing complete, sequence now: {final_sequence}")

        # Process X-site tokens
        x_tokens = site_tokens["x_tokens"]
        if x_tokens['Element']:
            x_token = [self.vocab.get(token, tok) for token in x_tokens["Element"]]
            self.logger.debug(f"X tokens: {x_tokens['Element']}, Encoded as: {x_token}")
            
            if len(x_tokens["Element"]) == 1:
                self.logger.debug(f"Single X-site element, using prop_x_1={row_data['prop_x_1']}, r_x={row_data['r_x']}")
                x_part = [[x_token, row_data['prop_x_1'], row_data['prop_x_2'], row_data['r_x']]]
                final_sequence.extend(x_part)
            elif len(x_tokens["Element"]) == 2:    
                self.logger.debug(f"Two X-site elements, using prop_x_1={row_data['prop_x_1']}, prop_x_2={row_data['prop_x_2']}, r_x={row_data['r_x']}")
                x_part = [[x_token, row_data['prop_x_1'], row_data['prop_x_2'], row_data['r_x']]]
                final_sequence.extend(x_part)
            self.logger.info(f"X-site processing complete, sequence now: {final_sequence}")
        
        # Add structure and tolerance factor
        final_sequence.append(row_data["structure"])
        final_sequence.append(row_data["tf"])
        self.logger.debug(f"Added structure ({row_data['structure']}) and tolerance factor ({row_data['tf']}) to sequence")
        
        self.logger.debug(f"Step '_build_sequence' completed in {time() - step_start:.4f} seconds")
        return final_sequence
    
    def _calculate_charge_balance(self, site_tokens):
        """Calculate charge balance for the chemical formula."""
        step_start = time()
        self.logger.debug("Calculating charge balance")
        
        # Combine all elements and probabilities
        all_elements = (site_tokens["a_tokens"]["Element"] + 
                         site_tokens["b_tokens"]["Element"] + 
                         site_tokens["x_tokens"]["Element"])
        all_probs = (site_tokens["a_tokens"]["Probs"] + 
                      site_tokens["b_tokens"]["Probs"] + 
                      site_tokens["x_tokens"]["Probs"])
        
        self.logger.debug(f"Combined elements list: {all_elements}")
        self.logger.debug(f"Combined probabilities list: {all_probs}")
        
        total_charge = 0
        charge_contributions = []
        
        for element, prob in zip(all_elements, all_probs):
            if element in self.element_charges:
                element_charge = self.element_charges[element]
                site = self.element_sites[element]
                contribution = prob * element_charge
                total_charge += contribution
                charge_contributions.append((element, prob, element_charge, contribution, site))
                self.logger.debug(f"Element: {element} ({site}), Prob: {prob}, Charge: {element_charge}, Contribution: {contribution}, Running total: {total_charge}")
            else:
                self.logger.warning(f"No charge found for element {element}")
        
        # Log summary of charge calculation
        self.logger.info("Charge balance calculation:")
        for element, prob, charge_val, contribution, site in charge_contributions:
            self.logger.info(f"  {element} ({site}): {prob} × {charge_val} = {contribution}")
        self.logger.info(f"Total charge: {total_charge}")
        
        charge_balanced = abs(total_charge) <= 0.01  # Allow small floating point errors
        if not charge_balanced:
            self.logger.warning(f"Formula may not be charge-balanced: total charge = {total_charge}")
        else:
            self.logger.info("Formula appears to be charge-balanced")
        
        self.logger.debug(f"Step '_calculate_charge_balance' completed in {time() - step_start:.4f} seconds")
        
        return {
            "total_charge": total_charge,
            "is_balanced": charge_balanced,
            "contributions": charge_contributions
        }
    
    def _convert_to_tensor(self, final_sequence, bandgap):
        """Convert the final sequence to tensor representation."""
        step_start = time()
        self.logger.debug("Converting to tensor")
        
        try:
            # Flatten the nested sequence
            flat_seq = self._flatten_nested(final_sequence)
            self.logger.debug(f"Flattened sequence length: {len(flat_seq)}")
            self.logger.debug(f"Flattened values: {flat_seq}")
            
            # Convert to tensors
            torch_tensor = torch.FloatTensor(flat_seq)
            target = torch.tensor(bandgap)
            
            self.logger.info(f"Created tensor of shape {torch_tensor.shape}")
            self.logger.info(f"Target value (bandgap): {target}")
            
            self.logger.debug(f"Step '_convert_to_tensor' completed in {time() - step_start:.4f} seconds")
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
        for row_index in range(2):
            logger.info(f"Processing example row {row_index}")
            example = df.iloc[row_index]
            
            logger.info(f"Processing formula: {example['Name']}")
            X_, target = tokenizer.encode(example)

            logger.info(f"Result tensor X part {X_.tolist()}")        
            logger.info(f"Result tensor shape: {X_.shape}")
            logger.info(f"Target value (bandgap): {target}")
            
            end_time = time()
            logger.info(f"Total execution time: {end_time - start_time:.4f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)