import torch
import torch.nn as nn
import torch.nn.functional as F

class PowerGridLoss(nn.Module):
    """
    Combined loss function for power grid model predictions.
    Combines MSE loss for rho values and KL divergence loss for soft labels.
    """
    def __init__(self, rho_weight=0.7, soft_label_weight=0.3):
        """
        Initialize the power grid loss function.
        
        Args:
            rho_weight: Weight for rho value loss
            soft_label_weight: Weight for soft label loss
        """
        super(PowerGridLoss, self).__init__()
        self.rho_weight = rho_weight
        self.soft_label_weight = soft_label_weight
        
        # MSE loss for rho values
        self.rho_criterion = nn.MSELoss(reduction='none')
        
        # Small epsilon to avoid log(0)
        self.eps = 1e-8
    
    def forward(self, predictions, targets, action_weights=None):
        """
        Calculate the combined loss.
        
        Args:
            predictions: Dictionary containing:
                - 'rho_values': Predicted rho values of shape (batch_size, num_actions)
                - 'soft_labels': Predicted soft labels of shape (batch_size, num_actions)
            targets: Dictionary containing:
                - 'rho_values': Ground truth rho values of shape (batch_size, num_actions)
                - 'soft_labels': Ground truth soft labels of shape (batch_size, num_actions)
            action_weights: Optional tensor of action weights for each action (batch_size, num_actions)
                
        Returns:
            Dictionary containing:
                - 'loss': Combined loss value
                - 'rho_loss': MSE loss for rho values
                - 'soft_label_loss': KL divergence loss for soft labels
        """
        # Extract predictions and targets
        pred_rho = predictions['rho_values']
        pred_soft = predictions['soft_labels']
        target_rho = targets['rho_values']
        target_soft = targets['soft_labels']
        
        # Calculate MSE loss for rho values
        rho_loss = self.rho_criterion(pred_rho, target_rho)
        
        # Apply action_weights if provided
        if action_weights is not None:
            # Ensure action_weights is on the same device as rho_loss
            action_weights = action_weights.to(rho_loss.device)
            # Element-wise multiplication with action weights
            rho_loss = rho_loss * action_weights
        
        # Average the loss across the actions
        rho_loss = rho_loss.mean()
        
        # Calculate KL divergence loss for soft labels
        # Add small epsilon to avoid log(0)
        pred_soft = torch.clamp(pred_soft, min=self.eps, max=1.0)
        target_soft = torch.clamp(target_soft, min=self.eps, max=1.0)
        
        kl_loss = F.kl_div(
            torch.log(pred_soft),
            target_soft,
            reduction='batchmean'
        )
        
        # Combine losses with weights
        combined_loss = self.rho_weight * rho_loss + self.soft_label_weight * kl_loss
        
        return {
            'loss': combined_loss,
            'rho_loss': rho_loss,
            'soft_label_loss': kl_loss
        }