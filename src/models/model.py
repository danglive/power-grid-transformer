import torch
import torch.nn as nn
from .observation_encoder import ObservationEncoder
from .action_encoder import ActionEncoder
from .cross_attention import CrossAttention

class PowerGridModel(nn.Module):
    """
    Complete power grid model that integrates observation encoder, action encoder,
    and cross-attention to predict rho_max values and soft labels for actions.
    """
    def __init__(self, config):
        """
        Initialize the power grid model.
        
        Args:
            config: Dictionary containing model configuration
        """
        super(PowerGridModel, self).__init__()
        
        # Extract model parameters from config
        observation_dim = 3819  # Observation feature dimension
        action_dim = 1152       # Action vector dimension
        num_actions = 50        # Number of actions per sample
        
        # Get transformer parameters
        transformer_params = config.get('model_params', {}).get('transformer', {})
        d_model = transformer_params.get('d_model', 1024)
        num_layers = transformer_params.get('num_layers', 6)
        num_heads = transformer_params.get('num_heads', 8)
        dff = transformer_params.get('dff', 256)
        dropout_rate = transformer_params.get('dropout_rate', 0.1)
        
        # Create observation encoder
        self.observation_encoder = ObservationEncoder(
            input_dim=observation_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate
        )
        
        # Create action encoder
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            max_actions=num_actions
        )
        
        # Create cross-attention module
        self.cross_attention = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # Create prediction heads
        self.rho_head = nn.Linear(d_model, 1)  # For rho_max prediction
        self.soft_label_head = nn.Linear(d_model, 1)  # For soft_label prediction
        
        # Layer normalization for outputs
        self.rho_norm = nn.LayerNorm(1)
        self.softlabel_norm = nn.LayerNorm(1)
        
        # Activation for rho values (must be positive)
        self.rho_activation = nn.ReLU()
        
        # Softmax for batch normalization of soft labels
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, batch):
        """
        Forward pass for power grid model.
        
        Args:
            batch: Dictionary containing:
                - 'observation': Tensor of shape (batch_size, observation_dim)
                - 'action_vectors': Tensor of shape (batch_size, num_actions, action_dim)
                
        Returns:
            Dictionary containing:
                - 'rho_values': Predicted rho values of shape (batch_size, num_actions)
                - 'soft_labels': Predicted soft labels of shape (batch_size, num_actions)
        """
        # Extract inputs from batch
        observations = batch['observation']
        action_vectors = batch['action_vectors']
        
        # Encode observations
        obs_embedding = self.observation_encoder(observations)  # [batch_size, d_model]
        
        # Encode actions
        action_embeddings = self.action_encoder(action_vectors)  # [batch_size, num_actions, d_model]
        
        # Apply cross-attention
        contextualized_embeddings = self.cross_attention(obs_embedding, action_embeddings)  # [batch_size, num_actions, d_model]
        
        # Predict rho values
        rho_values = self.rho_head(contextualized_embeddings)  # [batch_size, num_actions, 1]
        rho_values = self.rho_norm(rho_values)
        rho_values = self.rho_activation(rho_values)  # Ensure positive values
        rho_values = rho_values.squeeze(-1)  # [batch_size, num_actions]
        
        # Predict soft labels
        soft_logits = self.soft_label_head(contextualized_embeddings)  # [batch_size, num_actions, 1]
        soft_logits = self.softlabel_norm(soft_logits)
        soft_logits = soft_logits.squeeze(-1)  # [batch_size, num_actions]
        
        # Apply softmax to normalize the soft labels
        soft_labels = self.softmax(soft_logits)  # [batch_size, num_actions]
        
        return {
            'rho_values': rho_values,
            'soft_labels': soft_labels
        }
    
    def predict(self, batch):
        """
        Make predictions and return additional information for evaluation.
        
        Args:
            batch: Dictionary containing input data
            
        Returns:
            Dictionary containing predictions and additional info
        """
        # Get model predictions
        predictions = self.forward(batch)
        
        # Get top-k actions based on predicted soft labels
        k_values = [1, 3, 5]  # Common top-k values to evaluate
        top_k_actions = {}
        
        for k in k_values:
            # Get indices of top-k soft labels
            _, top_k_indices = torch.topk(predictions['soft_labels'], k, dim=1)
            top_k_actions[f'top_{k}'] = top_k_indices
        
        # Add top-k actions to predictions
        predictions['top_k_actions'] = top_k_actions
        
        return predictions