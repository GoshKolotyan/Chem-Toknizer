import torch
import torch.nn as nn
from tokenizer import ChemTokenizer

class ChemTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1):
        super(ChemTransformer, self).__init__()
        
        # First, project the flat features to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Convert to sequence form with proper dimensions for transformer
        # For multi-head attention, hidden_dim must be divisible by num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers for predicting bandgap
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, x):
        # Add batch dimension if not present
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Project to hidden dimension - shape becomes [batch, hidden_dim]
        x = self.input_projection(x)
        
        # Add sequence dimension - shape becomes [batch, 1, hidden_dim]
        # This creates a sequence of length 1
        x = x.unsqueeze(1)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension - shape becomes [batch, hidden_dim]
        x = x.squeeze(1)
        
        # Output projection to get bandgap prediction
        x = self.output_layers(x)
        
        return x.squeeze(-1)

# Test the model
if __name__ == "__main__":
    # Initialize tokenizer
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