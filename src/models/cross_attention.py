import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    """
    Cross-attention module for power grid model.
    Uses observation embedding as key/value and action embeddings as query.
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        """
        Initialize the cross-attention module.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super(CrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        # Depth of each head
        self.depth = d_model // self.num_heads
        
        # Linear projections for Q, K, V
        self.wq = nn.Linear(d_model, d_model)  # For action embeddings (query)
        self.wk = nn.Linear(d_model, d_model)  # For observation embedding (key)
        self.wv = nn.Linear(d_model, d_model)  # For observation embedding (value)
        
        # Final output projection
        self.dense = nn.Linear(d_model, d_model)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result to shape (batch_size, num_heads, seq_len, depth).
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, obs_embedding, action_embeddings):
        """
        Forward pass for cross-attention.
        
        Args:
            obs_embedding: Observation embedding tensor of shape (batch_size, d_model)
            action_embeddings: Action embeddings tensor of shape (batch_size, num_actions, d_model)
            
        Returns:
            Output tensor of shape (batch_size, num_actions, d_model)
        """
        batch_size = action_embeddings.size(0)
        num_actions = action_embeddings.size(1)
        
        # Expand observation embedding to match action sequence length
        obs_expanded = obs_embedding.unsqueeze(1).expand(-1, num_actions, -1)  # [batch_size, num_actions, d_model]
        
        # Linear projections
        q = self.wq(action_embeddings)  # [batch_size, num_actions, d_model]
        k = self.wk(obs_expanded)       # [batch_size, num_actions, d_model]
        v = self.wv(obs_expanded)       # [batch_size, num_actions, d_model]
        
        # Split heads
        q = self.split_heads(q, batch_size)  # [batch_size, num_heads, num_actions, depth]
        k = self.split_heads(k, batch_size)  # [batch_size, num_heads, num_actions, depth]
        v = self.split_heads(v, batch_size)  # [batch_size, num_heads, num_actions, depth]
        
        # Scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, num_actions, num_actions]
        
        # Scale by sqrt(dk)
        scaled_attention_logits = matmul_qk / math.sqrt(self.depth)
        
        # Apply softmax to get attention weights
        attention_weights = nn.functional.softmax(scaled_attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, v)  # [batch_size, num_heads, num_actions, depth]
        
        # Reshape back
        output = output.permute(0, 2, 1, 3).contiguous()  # [batch_size, num_actions, num_heads, depth]
        output = output.view(batch_size, num_actions, self.d_model)  # [batch_size, num_actions, d_model]
        
        # Final projection
        output = self.dense(output)  # [batch_size, num_actions, d_model]
        
        # Add & norm (residual connection)
        output = self.layer_norm(action_embeddings + output)
        
        return output
