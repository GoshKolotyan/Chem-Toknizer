from tokenizer import ChemTokenizer
from transformer import ChemTransformer
import torch

torch.manual_seed(42)  # For reproducibility

tokenizer = ChemTokenizer()

# Example input
example = {
    "Name": ["Ba3CsGa5Se10Cl2"],
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

# Encode the example
X_, target = tokenizer.encode(example)
input_dim = X_.size(0)  # Get actual input dimension

# Initialize model with correct input dimension
model = ChemTransformer(input_dim=input_dim, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1)

# Process the example
output = model(X_)

print(f"Input shape: {X_.shape}")
print(f"Predicted bandgap: {output.item():.4f}")  
print(f"Actual bandgap: {target.item():.4f}")