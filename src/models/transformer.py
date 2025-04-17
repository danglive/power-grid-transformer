# src/models/transformer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Positional Encoding
# -----------------------------
def get_angles(pos, i, d_model):
    """
    Calculate the angles for the positional encoding.
    
    Args:
        pos: Position indices
        i: Dimension indices
        d_model: Model dimension
        
    Returns:
        Tensor of angles
    """
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.tensor(d_model, dtype=torch.float32))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """
    Create positional encoding matrix.
    
    Args:
        position: Maximum sequence length
        d_model: Model dimension
        
    Returns:
        Positional encoding of shape (1, position, d_model)
    """
    angle_rads = get_angles(
        torch.arange(position, dtype=torch.float32).unsqueeze(1),
        torch.arange(d_model, dtype=torch.float32).unsqueeze(0),
        d_model
    )
    
    # Apply sine to even indices and cosine to odd indices
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads.unsqueeze(0)  # (1, position, d_model)
    return pos_encoding

# -----------------------------
# Attention Modules
# -----------------------------
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate the attention weights and apply them to the values.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        mask: Optional mask for padding or lookahead
        
    Returns:
        Tuple of (output tensor, attention weights)
    """
    # Calculate query-key dot product
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    
    # Scale by sqrt(dk)
    dk = k.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(dk)
    
    # Apply mask if provided
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax to get attention weights
    attention_weights = nn.functional.softmax(scaled_attention_logits, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module that allows the model to jointly attend to 
    information from different representation subspaces.
    """
    def __init__(self, d_model, num_heads):
        """
        Initialize the multi-head attention module.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Ensure d_model is divisible by num_heads
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        # Depth of each head
        self.depth = d_model // self.num_heads
        
        # Linear projections for Q, K, V
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, v, k, q, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            v: Value tensor
            k: Key tensor
            q: Query tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size = q.size(0)
        
        # Linear projection and split into heads
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
        
        # Reshape to (batch_size, num_heads, seq_len, depth)
        q = q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        
        # Apply scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape and concatenate heads
        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.dense(scaled_attention)
        
        return output, attention_weights

class PointWiseFeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network, applied to each position separately and identically.
    Consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, d_model, dff):
        """
        Initialize the feed-forward network.
        
        Args:
            d_model: Model dimension
            dff: Hidden layer dimension
        """
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        """
        Forward pass for feed-forward network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.linear2(self.relu(self.linear1(x)))

class ReZero(nn.Module):
    """
    ReZero module that initializes a learnable parameter alpha to zero,
    which helps with training stability for deep networks.
    """
    def __init__(self):
        """Initialize the ReZero module with alpha parameter set to zero."""
        super(ReZero, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Forward pass for ReZero module.
        
        Args:
            x: Input tensor
            
        Returns:
            Scaled input tensor by alpha
        """
        return self.alpha * x

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer consisting of multi-head attention, feed-forward networks,
    LayerNorm, and ReZero connections.
    """
    def __init__(self, d_model, num_heads, dff):
        """
        Initialize the encoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden dimension of feed-forward network
        """
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn0 = PointWiseFeedForwardNetwork(d_model, dff)
        self.ffn1 = PointWiseFeedForwardNetwork(d_model, dff)
        self.rz0 = ReZero()
        self.rz1 = ReZero()
        self.rz2 = ReZero()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass for encoder layer.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        # First feed-forward network followed by ReZero and LayerNorm
        ffn_output = self.ffn0(x)
        out0 = x + self.rz0(ffn_output)
        out0 = self.layernorm1(out0)

        # Self-attention followed by ReZero and LayerNorm
        attn_output, attention_weights = self.mha(out0, out0, out0, mask)
        out1 = out0 + self.rz1(attn_output)
        out1 = self.layernorm2(out1)

        # Second feed-forward network followed by ReZero and LayerNorm
        ffn_output = self.ffn1(out1)
        out2 = out1 + self.rz2(ffn_output)
        out2 = self.layernorm3(out2)

        return out2, attention_weights

# -----------------------------
# TransformerEncoder
# -----------------------------
class TransformerEncoder(nn.Module):
    """
    Transformer encoder for vector sequences.
    Processes input features through transformer encoder layers and applies
    mean pooling to get a fixed-size feature vector.
    """
    def __init__(self, input_feature, num_layers, d_model, num_heads, dff, dropout_rate=0.1, max_position=5000):
        """
        Initialize the vector encoder.
        
        Args:
            input_feature: Input feature dimension
            num_layers: Number of encoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Hidden dimension of feed-forward network
            dropout_rate: Dropout rate
            max_position: Maximum sequence position
        """
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        
        # Embedding layer to project input features to model dimension
        self.embedding = nn.Linear(input_feature, d_model)
        
        # Create positional encoding
        pos_encoding = positional_encoding(max_position, d_model)
        self.register_buffer('pos_encoding_buffer', pos_encoding)
        
        # Encoder layers
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        """
        Forward pass for vector encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_feature)
            mask: Optional attention mask
            
        Returns:
            Pooled output tensor of shape (batch_size, d_model)
        """
        # Project input to model dimension
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding_buffer[:, :x.size(1), :].to(x.device)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.enc_layers:
            x, _ = layer(x, mask)
        
        # Mean pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        return x
