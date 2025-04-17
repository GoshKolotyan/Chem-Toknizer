import os
import re
import csv
import json
import torch
import logging
import pandas as pd

from time import time
from helper import ParamsCalcluator

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
            with open('../Data/canonical_vocab.json', 'r') as f:
                self.vocab = json.load(f)
            self.logger.info(f"Vocabulary loaded with {len(self.vocab)} entries")
        except Exception as e:
            self.logger.error(f"Failed to load vocabulary: {e}")
            raise
        
        # Load element data
        try:
            self.known_A_elements = pd.read_csv('../Data/A_part_AA.csv', delimiter=';')
            self.known_B_elements = pd.read_csv('../Data/B_part_AA.csv', delimiter=';')
            self.known_X_elements = pd.read_csv('../Data/X_part_AA.csv', delimiter=';')
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
            self.logger.info(f"Tokenized '{formula_str}' ReGexed is {raw_tokens}  into {len(raw_tokens)} tokens ")
            
            # Process tokens
            element_probs = self._pair_elements_with_probs(raw_tokens)
            X_finall_charge = None
            self.logger.info(f"After _pair_elements_with_probs function {element_probs}")
            if element_probs[-1][0].isdigit():
                X_finall_charge = int(element_probs[-1][0])
                self.logger.info(f"Finall Prob detected moved to -->{X_finall_charge}")
                element_probs.pop(-1)
            
            # Categorize tokens
            site_tokens = self._categorize_elements(element_probs)
            if site_tokens is None:
                self.logger.warning(f"Could not categorize elements for formula: {formula_str}")
                return None
            
            self.logger.info(f"After _categorize_elements function {site_tokens}")
                
            self.logger.info(f"Site Tokens A: {site_tokens['a_tokens']['Element']} {site_tokens['a_tokens']['Probs']}")
            self.logger.info(f"Site Tokens B: {site_tokens['b_tokens']['Element']} {site_tokens['b_tokens']['Probs']}")
            self.logger.info(f"Site Tokens X: {site_tokens['x_tokens']['Element']} {site_tokens['x_tokens']['Probs']}")

            
            # # Check if we have elements in all required sites
            if not site_tokens['a_tokens']['Element'] or not site_tokens['b_tokens']['Element'] or not site_tokens['x_tokens']['Element']:
                self.logger.warning(f"Missing elements in one or more sites for formula: {formula_str}")
                return None
            
            # # Prepare input for ParamsCalculator - ensure each site has exactly what it needs
            A_part = site_tokens['a_tokens']["Element"] + site_tokens['a_tokens']["Probs"]
            B_part = site_tokens['b_tokens']["Element"] + site_tokens['b_tokens']["Probs"]
            X_part = site_tokens['x_tokens']["Element"] + site_tokens['x_tokens']["Probs"]
            
            # # Make sure we have exactly 4 elements in each part (element1, element2, prob1, prob2)
            if len(A_part) != 4 or len(B_part) != 4 or len(X_part) != 4:
                self.logger.warning(f"Incorrect element format for ParamsCalculator. A:{len(A_part)}, B:{len(B_part)}, X:{len(X_part)}")
                # Pad if needed
                while len(A_part) < 4:
                    if len(A_part) % 2 == 0:  # Need element
                        A_part.insert(len(A_part) // 2, "")
                    else:  # Need probability
                        A_part.append(0.0)
                        
                while len(B_part) < 4:
                    if len(B_part) % 2 == 0:
                        B_part.insert(len(B_part) // 2, "")
                    else:
                        B_part.append(0.0)
                        
                while len(X_part) < 4:
                    if len(X_part) % 2 == 0:
                        X_part.insert(len(X_part) // 2, "")
                    else:
                        X_part.append(0.0)
            
            # Calculate parameters using ParamsCalculator
            calculator = ParamsCalcluator(A_part=A_part, B_part=B_part, X_part=X_part, x_backup = X_finall_charge)
            result = calculator()
            
            if result is None:
                self.logger.warning(f"Charge not balanced for formula: {formula_str}")
                return None
            
            A_radius, B_radius, X_radius, tolerance_factor, structure = result
            
            self.logger.info(f"Calculated parameters: A_radius={A_radius:.4f}, B_radius={B_radius:.4f}, "
                           f"X_radius={X_radius:.4f}, tolerance_factor={tolerance_factor:.4f}, "
                           f"structure={structure}")
            
            # Add calculated parameters to row_data for tensor creation
            row_data.update({
                "r_a": A_radius,
                "r_b": B_radius,
                "r_x": X_radius,
                "tf": tolerance_factor,
                "structure": structure,
                "prop_a_1": site_tokens['a_tokens']["Probs"][0],
                "prop_a_2": site_tokens['a_tokens']["Probs"][1],
                "prop_b_1": site_tokens['b_tokens']["Probs"][0],
                "prop_b_2": site_tokens['b_tokens']["Probs"][1],
                "prop_b_3": 0,  # Default value if not used
                "prop_x_1": site_tokens['x_tokens']["Probs"][0],
                "prop_x_2": site_tokens['x_tokens']["Probs"][1]
            })
            
            # Build sequence for tensor
            final_sequence = self._build_sequence(site_tokens, row_data)
            
            # Convert to tensor
            tensor_result = self._convert_to_tensor(final_sequence, row_data["bandgap"])
            
            return tensor_result
            
        except Exception as e:
            self.logger.error(f"Error during encoding: {e}", exc_info=True)
            raise
    
    def _extract_row_data(self, row):
        """Extract data from the input row and handle missing values."""
        try:
            # Get required fields with defaults for missing values
            row_data = {
                "name_value": row["Name"],
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

        # Filter out numerical values that are not associated with elements
        # (this handles cases where numbers like "3" appear at the end of formulas)
        filtered_element_probs = []
        for element, prob in element_probs:
            try:
                # Skip if the element is just a number
                if element.isdigit() or (element.replace('.', '', 1).isdigit() and element.count('.') <= 1):
                    self.logger.warning(f"Skipping standalone number: {element}")
                    continue
                filtered_element_probs.append((element, prob))
            except:
                filtered_element_probs.append((element, prob))
        self.logger.info(f"filtered_element_probs is  ---> {filtered_element_probs}")

        # First pass: categorize elements by site
        for element, prob in filtered_element_probs:
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
        
        self.logger.info(f"Site tokens 1 phase is  ---> {site_tokens}")

        # Second pass: normalize probabilities for each site
        for site_key in ["a_tokens", "b_tokens", "x_tokens"]:
            site = site_tokens[site_key]
            if not site["Element"]:  # Skip empty sites
                continue
                
            # total_prob = sum(site["Probs"])
            # if total_prob > 0 and abs(total_prob - 1.0) > 0.001:
            #     site["Probs"] = [p/total_prob for p in site["Probs"]]
        self.logger.info(f"Site tokens 2 phase is  ---> {site_tokens}")

        # Third pass: handle cases where elements are found in multiple sites
        # If some elements are placed in the wrong sites, we need to fix them
        
        # Ensure we have at least one element in each site
        missing_sites = []
        if not site_tokens["a_tokens"]["Element"]:
            missing_sites.append("a_tokens")
        if not site_tokens["b_tokens"]["Element"]:
            missing_sites.append("b_tokens") 
        if not site_tokens["x_tokens"]["Element"]:
            missing_sites.append("x_tokens")
            
        # If any site is missing, we need to ensure we have the complete ABX3 structure
        # For now, we'll handle common cases like CsInS2 (which should be Cs on A-site, In on B-site, S on X-site)
        if missing_sites:
            # Special case handling - we need A, B, and X sites for perovskites
            self.logger.warning(f"Missing elements for sites: {missing_sites}")
            
            # Add empty elements to missing sites with default values
            for site in missing_sites:
                site_tokens[site]["Element"] = ["", ""]
                site_tokens[site]["Probs"] = [1.0, 0.0]
        
        # Make sure we handle incorrect site assignments - e.g., if In appears in A-site when it should be B-site
        # This requires domain knowledge, but for now we'll just ensure we have the right structure
        
        # Fourth pass: ensure each site has exactly 2 elements (for double perovskites)
        for site_key in ["a_tokens", "b_tokens", "x_tokens"]:
            site = site_tokens[site_key]
            
            # If we have only one element, add an empty one
            if len(site["Element"]) == 1:
                site["Element"].append("")
                site["Probs"].append(0.0)
            # If we have more than 2 elements, keep only the two with highest probabilities
            elif len(site["Element"]) > 2:
                combined = list(zip(site["Element"], site["Probs"]))
                combined.sort(key=lambda x: x[1], reverse=True)
                combined = combined[:2]
                
                # Normalize probabilities of the two selected elements
                total = sum(p for _, p in combined)
                if total > 0:
                    site["Element"] = [e for e, _ in combined]
                    site["Probs"] = [p/total for _, p in combined]
                else:
                    site["Element"] = [e for e, _ in combined]
                    site["Probs"] = [1.0, 0.0]  # Default if all probs are 0
            # If no elements, add two empty ones
            elif len(site["Element"]) == 0:
                site["Element"] = ["", ""]
                site["Probs"] = [1.0, 0.0]
        
        return site_tokens
    
    def _build_sequence(self, site_tokens, row_data):
        """Build the sequence from categorized tokens and additional properties."""
        final_sequence = []
        tok = 0  # Default token ID if not found in vocabulary

        # Process A-site tokens
        a_tokens = site_tokens["a_tokens"]
        if a_tokens['Element']:
            a_token = [self.vocab.get(token, tok) for token in a_tokens['Element']]
            a_part = [a_token, row_data['prop_a_1'], row_data['prop_a_2'], row_data['r_a']]
            final_sequence.append(a_part)

        # Process B-site tokens
        b_tokens = site_tokens["b_tokens"]
        if b_tokens['Element']:
            b_token = [self.vocab.get(token, tok) for token in b_tokens["Element"]]
            b_part = [b_token, row_data['prop_b_1'], row_data['prop_b_2'], row_data['r_b']]
            final_sequence.append(b_part)

        # Process X-site tokens
        x_tokens = site_tokens["x_tokens"]
        if x_tokens['Element']:
            x_token = [self.vocab.get(token, tok) for token in x_tokens["Element"]]
            x_part = [x_token, row_data['prop_x_1'], row_data['prop_x_2'], row_data['r_x']]
            final_sequence.append(x_part)
        
        # Add structure and tolerance factor
        final_sequence.append(row_data["structure"])
        final_sequence.append(row_data["tf"])
        
        return final_sequence
    
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
        df = pd.read_csv("../Cleaned Data/Full_Data_ranamed_columns.csv")
        logger.info(f"Loaded data with {len(df)} rows")
        
        tokenizer = ChemTokenizer()
        valid_count = 0
        invalid_count = 0
        
        # Create a CSV file to store invalid formulas
        balance_checker_file = "balance_checker.csv"
        with open(balance_checker_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(['Invalid_Formula', 'Reason'])
            
            # Process example rows
            for row_index in range(len(df)):
                example = df.iloc[row_index]
                logger.info(f"Processing formula: {example['Name']}")
                result = tokenizer.encode(example)
                
                if result is None:
                    reason = "Charge imbalance or categorization issues"
                    logger.warning(f"Skipped formula {example['Name']} due to {reason}")
                    # Write invalid formula to CSV
                    csv_writer.writerow([example['Name'], reason])
                    invalid_count += 1
                else:
                    X_, target = result
                    logger.info(f"Result tensor shape: {X_.shape}, Target value: {target}")
                    valid_count += 1
                # break
        
        logger.info(f"Processing complete. Valid formulas: {valid_count}, Invalid formulas: {invalid_count}")
        logger.info(f"Invalid formulas have been saved to {os.path.abspath(balance_checker_file)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)