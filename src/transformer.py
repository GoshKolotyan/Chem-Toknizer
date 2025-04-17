import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PositionalEncoding(nn.Module):
    """
    Custom positional encoding for chemical data that respects
    the structure of perovskite feature vectors.
    """
    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encodings matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
        Returns:
            Output tensor with added positional encoding
        """
        return x + self.pe[:, :x.size(1), :]

class ChemicalEmbedding(nn.Module):
    """
    Embedding layer for chemical features that adapts the raw feature vectors
    to the size expected by the Transformer.
    """
    def __init__(self, input_dim, embedding_dim):
        super(ChemicalEmbedding, self).__init__()
        self.embedding = nn.Linear(1, embedding_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len]
        Returns:
            Embedded tensor of shape [batch_size, seq_len, embedding_dim]
        """
        # Reshape input to [batch_size, seq_len, 1]
        x = x.unsqueeze(-1)
        # Apply embedding to each feature individually
        return self.embedding(x)

class ChemTransformer(nn.Module):
    """
    Lightweight Transformer model for chemical property prediction.
    Combines a small Transformer encoder with a traditional ML regression head.
    """
    def __init__(
        self, 
        input_dim=18,         # Default for your perovskite feature vector length
        embedding_dim=32,     # Smaller embedding dimension
        num_heads=4,          # Reduced number of attention heads
        num_layers=2,         # Only 2 Transformer encoder layers
        dim_feedforward=64,   # Smaller feedforward layer
        dropout=0.1,
        activation="relu"
    ):
        super(ChemTransformer, self).__init__()
        
        # Input embedding to convert chemical features to transformer dimensions
        self.embedding = ChemicalEmbedding(input_dim, embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # Use batch_first=True for PyTorch ≥ 1.9.0
        )
        
        # Lightweight transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Traditional ML regression head
        self.regression_head = nn.Sequential(
            nn.Linear(embedding_dim * input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len]
        Returns:
            Output tensor of shape [batch_size, 1] (bandgap prediction)
        """
        # Embed input features
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Flatten for regression head
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        
        # Apply regression head
        x = self.regression_head(x)
        
        return x.squeeze(-1)  # Return [batch_size] tensor
