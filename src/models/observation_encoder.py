import torch
import torch.nn as nn
from .transformer import EncoderLayer

class ObservationEncoder(nn.Module):
    """
    Encoder for power grid observation vectors.
    Processes observation features through transformer encoder layers.
    """
    def __init__(self, input_dim, d_model, num_layers, num_heads, dff, dropout_rate=0.1):
        """
        Initialize the observation encoder.
        
        Args:
            input_dim: Dimension of the observation vector (typically 3819)
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            dff: Hidden dimension of feed-forward network
            dropout_rate: Dropout rate
        """
        super(ObservationEncoder, self).__init__()
        
        # Projection layer to map observation vector to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        """
        Forward pass for observation encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Encoded observation tensor of shape (batch_size, d_model)
        """
        # Project input to model dimension
        x = self.input_projection(x)  # [batch_size, d_model]
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Add dummy sequence dimension for encoder layers
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x, _ = layer(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, d_model]
        
        return x
