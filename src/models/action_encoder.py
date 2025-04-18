import torch
import torch.nn as nn
from .transformer import EncoderLayer, positional_encoding

class ActionEncoder(nn.Module):
    """
    Encoder for power grid action vectors.
    Processes sequence of action vectors through transformer encoder layers.
    """
    def __init__(self, action_dim, d_model, num_layers, num_heads, dff, dropout_rate=0.1, max_actions=50):
        """
        Initialize the action encoder.
        
        Args:
            action_dim: Dimension of each action vector (typically 1152)
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            dff: Hidden dimension of feed-forward network
            dropout_rate: Dropout rate
            max_actions: Maximum number of actions (typically 50)
        """
        super(ActionEncoder, self).__init__()
        
        # Projection layer to map action vectors to model dimension
        self.input_projection = nn.Linear(action_dim, d_model)
        
        # Create positional encoding
        pos_encoding = positional_encoding(max_actions, d_model)
        self.register_buffer('pos_encoding', pos_encoding)
        
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
        Forward pass for action encoder.
        
        Args:
            x: Input tensor of shape (batch_size, num_actions, action_dim)
            
        Returns:
            Encoded action tensor of shape (batch_size, num_actions, d_model)
        """
        # Project input to model dimension
        x = self.input_projection(x)  # [batch_size, num_actions, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x, _ = layer(x)
        
        return x  # [batch_size, num_actions, d_model]
