import numpy as np
import pandas as pd
from math import sqrt, log2
import logging

# Example usage
# This shows how to use the ParamsCalculator class to evaluate perovskite structures


class ParamsCalcluator:
    def __init__(self, A_part, B_part, X_part, x_backup=None):
        # Set up logger
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.A_composition = A_part
        self.B_composition = B_part
        self.X_composition = X_part
        self.X_backup = x_backup
        
        try:
            # Load data
            self.A = pd.read_csv("../Data/A_part_AA.csv", delimiter=";")
            self.B = pd.read_csv("../Data/B_part_AA.csv", delimiter=";")
            self.X = pd.read_csv("../Data/X_part_AA.csv", delimiter=";")
            
            # Create combined dataframes for lookup
            self.all_radiuses = pd.concat([
                self.A[["Element", "Ionic radius/Å"]],
                self.B[["Element", "Ionic radius/Å"]],
                self.X[["Element", "Ionic radius/Å"]]
            ])
            
            # Create a combined dataframe with elements and charges
            self.all_charges = pd.concat([
                self.A[['Element', "Charge"]], 
                self.B[['Element', "Charge"]], 
                self.X[['Element', "Charge"]]
            ])
            
            # Create dictionaries for quick lookup
            self.element_charges = dict(zip(self.all_charges['Element'], self.all_charges['Charge']))
            self.element_radiuses = dict(zip(self.all_radiuses['Element'], self.all_radiuses["Ionic radius/Å"]))
            
            # self.logger.info(f"Successfully loaded data: {len(self.element_charges)} elements")
            
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def global_radiuses(self, element_array):
        """
        Calculate weighted average radius based on element composition.
        
        Parameters:
        -----------
        element_array : list
            [element1, element2, prob1, prob2] - List containing elements and their probabilities
            
        Returns:
        --------
        float
            Weighted average ionic radius
        """
        element_1, element_2, prob_1, prob_2 = element_array
        
        # Look up radii in the dictionary
        element_1_radius = self.element_radiuses.get(element_1, 0)
        if element_1_radius == 0 and element_1:
            self.logger.warning(f"Radius for element {element_1} not found, using 0")
            
        element_2_radius = self.element_radiuses.get(element_2, 0)
        if element_2_radius == 0 and element_2:
            self.logger.warning(f"Radius for element {element_2} not found, using 0")
        
        # Calculate weighted average
        weighted_radius = element_1_radius * prob_1
        if element_2 and prob_2 > 0:
            weighted_radius += element_2_radius * prob_2
            
        return weighted_radius

    def tolerance_factor(self, R_a: float, R_b: float, R_x: float):
        """
        Calculate Goldschmidt tolerance factor.
        
        Parameters:
        -----------
        R_a : float
            A-site cation radius
        R_b : float
            B-site cation radius
        R_x : float
            X-site anion radius
            
        Returns:
        --------
        float
            Tolerance factor
        """
        # Avoid division by zero
        denominator = sqrt(2) * (R_x + R_b)
        if denominator == 0:
            self.logger.warning("Division by zero in tolerance_factor calculation")
            return 0  # Or return None or raise an exception
            
        return (R_a + R_x) / denominator

    def tolerance_factor_new(self, R_a: float, R_b: float, R_x: float, n_a: any):
        """Alternative tolerance factor calculation"""
        # Avoid division by zero or log of negative numbers
        if R_b == 0 or R_x == 0 or R_a == 0 or R_a / R_x <= 0:
            self.logger.warning("Invalid values for tolerance_factor_new calculation")
            return 0
            
        return (R_x / R_b) - n_a * (n_a - (R_a / R_b) / log2(R_a / R_x))

    def calculate_structure(self, tolerance_factor):
        """
        Determine structure type based on tolerance factor.
        
        Parameters:
        -----------
        tolerance_factor : float
            Goldschmidt tolerance factor
            
        Returns:
        --------
        int
            Structure type (0-3)
            1: orthorhombic (0.8 <= t < 0.9)
            2: tetragonal (0.9 <= t < 1.0)
            3: cubic (1.0 <= t <= 1.2)
            0: non-perovskite (t < 0.8 or t > 1.2)
        """
        if 0.8 <= tolerance_factor < 0.9:
            return 1  # orthorhombic
        elif 0.9 <= tolerance_factor < 1:
            return 2  # tetragonal
        elif 1 <= tolerance_factor <= 1.2:
            return 3  # cubic
        else:
            return 0  # non-perovskite

    def calculate_charge(self, elements):
        """
        Calculate if the total charge is balanced for the given elements.
        
        Parameters:
        -----------
        elements : list
            List of elements and their stoichiometry, e.g., ["MA", "Pb", "Cl", 3]
            The last value represents the stoichiometry of the X element.
            
        Example:
        --------
        For MAPbI3 (methylammonium lead iodide):
            elements = ["MA", "Pb", "I", 3]
            
        For CsSnCl3 (cesium tin chloride):
            elements = ["Cs", "Sn", "Cl", 3]
        
        Returns:
        --------
        bool
            True if the charge is balanced, False otherwise.
        """
        if len(elements) < 3:
            self.logger.warning("Not enough elements provided to calculate charge")
            return False
            
        # Extract elements and stoichiometry
        A_element_1 = elements[0][0]
        A_element_2 = elements[0][1]
        A_element_1_prob = elements[0][2]
        A_element_2_prob = elements[0][3]
        logging.info(f"A_1 site elements is -----> {A_element_1} with prob {A_element_1_prob} A_2 site elements is -----> {A_element_2} with prob {A_element_2_prob}")
        B_element_1 = elements[1][0]  # B-site cation
        B_element_2 = elements[1][1]
        B_element_1_prob = elements[1][2]
        B_element_2_prob = elements[1][3]
        logging.info(f"B_1 site elements is -----> {B_element_1} with prob {B_element_1_prob} B_2 site elements is -----> {B_element_2} with prob {B_element_2_prob}")

        X_element_1 = elements[2][0]  # X-site anion
        X_element_2 = elements[2][1] 
        X_element_1_prob = elements[2][2]
        X_element_2_prob = elements[2][3]
        
            
        logging.info(f"X_1 site elements is -----> {X_element_1} with prob {X_element_1_prob} X_2 site elements is -----> {X_element_2} with prob {X_element_2_prob}")
        
        # Get charges for each element
        A_charge_1, A_charge_2 = self.element_charges.get(A_element_1, 0), self.element_charges.get(A_element_2, 0)
        B_charge_1, B_charge_2 = self.element_charges.get(B_element_1, 0), self.element_charges.get(B_element_2, 0)
        X_charge_1, X_charge_2 = self.element_charges.get(X_element_1, 0), self.element_charges.get(X_element_2, 0)
        
        # Check if all charges were found
        if A_charge_1 is None:
            # self.logger.warning(f"No charge found for A-site element {A_element}")
            return False
        if A_charge_2 is None:
            return False
        if B_charge_1 is None:
            # self.logger.warning(f"No charge found for B-site element {B_element}")
            return False
        if B_charge_2 is None:
            return False
        if X_charge_1 is None:
            # self.logger.warning(f"No charge found for X-site element {X_element}")
            return False
        if X_charge_2 is None:
            return False            
        # Calculate total charge: A + B + X*stoichiometry
        if self.X_backup:
            total_charge = (A_charge_1 * A_element_1_prob + A_charge_2 * A_element_2_prob) \
                        + (B_charge_1 * B_element_1_prob + B_charge_2 * B_element_2_prob) \
                        + (X_charge_1 * X_element_1_prob + X_charge_2 * X_element_2_prob)*self.X_backup
            self.logger.info(f"Total charge: {total_charge} A: {(A_charge_1 * A_element_1_prob + A_charge_2 * A_element_2_prob)} + B: {(B_charge_1 * B_element_1_prob + B_charge_2 * B_element_2_prob)} + X: {(X_charge_1 * X_element_1_prob + X_charge_2 * X_element_2_prob)*self.X_backup}")
        else:
            total_charge = (A_charge_1 * A_element_1_prob + A_charge_2 * A_element_2_prob) \
                        + (B_charge_1 * B_element_1_prob + B_charge_2 * B_element_2_prob) \
                        + (X_charge_1 * X_element_1_prob + X_charge_2 * X_element_2_prob)
            self.logger.info(f"Total charge: {total_charge} A: {(A_charge_1 * A_element_1_prob + A_charge_2 * A_element_2_prob)} + B: {(B_charge_1 * B_element_1_prob + B_charge_2 * B_element_2_prob)} + X: {(X_charge_1 * X_element_1_prob + X_charge_2 * X_element_2_prob) }")
        
        # Check if charge is balanced (should be close to zero)
        charge_balanced = abs(total_charge) <= 0.01  # Allow small floating point errors
        
        return charge_balanced, total_charge

    def __call__(self):
        """
        Calculate perovskite structural parameters.
        
        Returns:
        --------
        tuple or None
            (A_radius, B_radius, X_radius, tolerance_factor, structure) if charge is balanced
            None if charge is not balanced
        """
        try:
            # Calculate radiuses
            self.logger.info(f"A-composotion is --->{self.A_composition} B-compositions is --->{self.B_composition} X-composition is --->{self.X_composition}")
            A_radius = self.global_radiuses(self.A_composition)
            # self.logger.info(f"A-site radius: {A_radius}")
            
            B_radius = self.global_radiuses(self.B_composition)
            # self.logger.info(f"B-site radius: {B_radius}")
            
            X_radius = self.global_radiuses(self.X_composition)
            # self.logger.info(f"X-site radius: {X_radius}")
            
            # Calculate tolerance factor
            tolerance_factor = self.tolerance_factor(R_a=A_radius, R_b=B_radius, R_x=X_radius)
            # self.logger.info(f"Tolerance factor: {tolerance_factor}")
            
            # Determine structure type
            structure = self.calculate_structure(tolerance_factor=tolerance_factor)
            # self.logger.info(f"Structure type: {structure}")
            
            # Check charge balance
            elements = [self.A_composition, self.B_composition, self.X_composition]
            charge_balanced, total_charge = self.calculate_charge(elements)
            
            if charge_balanced:  # This is already a boolean value
                return A_radius, B_radius, X_radius, tolerance_factor, structure
            else:
                # self.logger.warning(f"Charge is {total_charge}, returning None")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in calculation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None