import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_action_metrics(predictions, targets, k_values=(1, 3, 5)):
    """Compute metrics for action selection based on soft labels."""
    # Extract predictions
    pred_soft = predictions['soft_labels']
    
    # Use target soft labels to evaluate
    target_soft = targets['soft_labels']
    
    # For each sample, get indices sorted by probability (ground truth)
    _, target_indices_sorted = torch.sort(target_soft, dim=1, descending=True)
    
    # Get the top-1 index for each sample as the best action
    target_best = target_indices_sorted[:, 0]
    
    # Convert to tensor if not already a tensor and ensure it's the correct type
    if not isinstance(target_best, torch.Tensor):
        target_best = torch.tensor(target_best, dtype=torch.long, device=pred_soft.device)
    
    # Initialize metrics
    metrics = {}
    
    # Compute metrics for each k
    for k in k_values:
        # Get top-k predicted actions
        _, pred_top_k = torch.topk(pred_soft, k, dim=1)
        
        # 1. Top-k Accuracy: Is the best action in top-k predictions?
        in_top_k = torch.zeros_like(target_best, dtype=torch.float32)
        for i in range(k):
            in_top_k += (pred_top_k[:, i] == target_best).float()
        top_k_acc = in_top_k.mean().item()
        metrics[f'top_{k}_accuracy'] = top_k_acc
        
        # 2. Recall@k: Fraction of relevant items retrieved
        if k > 1:
            # Get top-k ground truth actions
            top_k_gt = target_indices_sorted[:, :k]
            
            # Count matches between predictions and ground truth
            recall_sum = torch.zeros(len(pred_soft), dtype=torch.float32, device=pred_soft.device)
            for i in range(len(pred_soft)):
                # Convert tensors to sets for intersection
                pred_set = set(pred_top_k[i].cpu().numpy())
                gt_set = set(top_k_gt[i].cpu().numpy())
                # Compute recall: |relevant ∩ retrieved| / |relevant|
                recall_sum[i] = len(pred_set.intersection(gt_set)) / len(gt_set)
            
            metrics[f'recall@{k}'] = recall_sum.mean().item()
            
            # 3. Precision@k: Fraction of retrieved items that are relevant
            precision_sum = torch.zeros(len(pred_soft), dtype=torch.float32, device=pred_soft.device)
            for i in range(len(pred_soft)):
                pred_set = set(pred_top_k[i].cpu().numpy())
                gt_set = set(top_k_gt[i].cpu().numpy())
                # Compute precision: |relevant ∩ retrieved| / |retrieved|
                precision_sum[i] = len(pred_set.intersection(gt_set)) / len(pred_set)
            
            metrics[f'precision@{k}'] = precision_sum.mean().item()
            
            # 4. NDCG@k: Normalized Discounted Cumulative Gain
            ndcg_sum = torch.zeros(len(pred_soft), dtype=torch.float32, device=pred_soft.device)
            for i in range(len(pred_soft)):
                # Get relevance scores for predicted items (1 if in ground truth, 0 otherwise)
                relevance = torch.zeros(k, dtype=torch.float32, device=pred_soft.device)
                for j in range(k):
                    if pred_top_k[i, j] in top_k_gt[i]:
                        relevance[j] = 1.0
                
                # Compute DCG: sum(rel_i / log2(i+2))
                discount = torch.log2(torch.arange(2, k+2, dtype=torch.float32, device=pred_soft.device))
                dcg = torch.sum(relevance / discount)
                
                # Compute ideal DCG (perfect ranking)
                ideal_relevance = torch.ones(min(k, len(gt_set)), dtype=torch.float32, device=pred_soft.device)
                ideal_dcg = torch.sum(ideal_relevance / discount[:min(k, len(gt_set))])
                
                # Compute NDCG
                if ideal_dcg > 0:
                    ndcg_sum[i] = dcg / ideal_dcg
                else:
                    ndcg_sum[i] = 0.0
            
            metrics[f'ndcg@{k}'] = ndcg_sum.mean().item()
    
    return metrics

def compute_rho_metrics(predictions, targets):
    """Compute comprehensive metrics for rho value predictions."""
    # Extract predictions and targets
    pred_rho = predictions['rho_values']
    target_rho = targets['rho_values']
    
    # Compute mean absolute error
    mae = torch.abs(pred_rho - target_rho).mean().item()
    
    # Compute mean squared error
    mse = ((pred_rho - target_rho) ** 2).mean().item()
    
    # Compute root mean squared error
    rmse = torch.sqrt(torch.mean((pred_rho - target_rho) ** 2)).item()
    
    # Compute R-squared for each action
    target_mean = target_rho.mean(dim=0, keepdim=True)
    ss_total = ((target_rho - target_mean) ** 2).sum(dim=0)
    ss_residual = ((target_rho - pred_rho) ** 2).sum(dim=0)
    r_squared = 1 - (ss_residual / (ss_total + 1e-8))
    mean_r_squared = r_squared.mean().item()
    
    # Compute Pearson correlation
    pred_mean = pred_rho.mean(dim=0, keepdim=True)
    numerator = ((pred_rho - pred_mean) * (target_rho - target_mean)).sum(dim=0)
    denominator = torch.sqrt(((pred_rho - pred_mean) ** 2).sum(dim=0) * ((target_rho - target_mean) ** 2).sum(dim=0) + 1e-8)
    pearson_corr = numerator / denominator
    mean_pearson = pearson_corr.mean().item()
    
    # Compute Spearman correlation (approximation using ranks)
    # Convert to ranks per sample
    batch_size = pred_rho.size(0)
    spearman_sum = 0.0
    
    # Compute for each sample separately (across actions)
    for i in range(batch_size):
        # Get ranks for prediction and target
        pred_ranks = torch.argsort(torch.argsort(pred_rho[i])).float()
        target_ranks = torch.argsort(torch.argsort(target_rho[i])).float()
        
        # Compute squared difference of ranks
        d_squared = torch.sum((pred_ranks - target_ranks) ** 2)
        
        # Spearman formula: 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
        n = float(pred_rho.size(1))
        spearman = 1.0 - (6.0 * d_squared) / (n * (n**2 - 1.0))
        spearman_sum += spearman.item()
    
    mean_spearman = spearman_sum / batch_size
    
    return {
        'rho_mae': mae,
        'rho_mse': mse,
        'rho_rmse': rmse,
        'rho_r_squared': mean_r_squared,
        'rho_pearson': mean_pearson,
        'rho_spearman': mean_spearman
    }

def compute_all_metrics(predictions, targets):
    """Compute all metrics for model evaluation."""
    action_metrics = compute_action_metrics(predictions, targets)
    rho_metrics = compute_rho_metrics(predictions, targets)
    
    # Example of combining metrics into one dictionary
    all_metrics = {**action_metrics, **rho_metrics}
    
    return all_metrics