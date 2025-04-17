"""
Enhanced data splitting strategies with comprehensive drift monitoring.
Provides advanced techniques for data splitting, visualization, and drift detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
import logging
from datetime import datetime
from pathlib import Path
import json
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedSplitManager:
    """
    Enhanced split manager with drift monitoring capabilities and multiple splitting strategies.
    Provides comprehensive analysis and visualization of data splits.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        visualization_dir: Optional[str] = None
    ):
        """
        Initialize the enhanced split manager.
        
        Args:
            config: Configuration dictionary
            visualization_dir: Directory for saving visualizations
        """
        self.config = config
        self.val_size = config.get('val_size', 0.2)
        self.test_size = config.get('test_size', 0.0)
        self.split_strategy = config.get('split_strategy', 'random')
        self.random_seed = config.get('random_seed', 42)
        self.stratify_key = config.get('stratify_key', None)
        self.gap_days = config.get('gap_days', 0)  # Days gap between train and validation
        
        # Set visualization directory
        self.visualization_dir = visualization_dir or config.get('visualization_dir', 'data_visualizations')
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Initialize split statistics
        self.split_stats = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        # Initialize drift metrics
        self.drift_metrics = {}
        self.action_drift_metrics = {}
        
        # Initialize random number generator
        self.rng = np.random.RandomState(self.random_seed)
    
    def split_data(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices, test indices)
        """
        total_samples = len(data['obs'])
        indices = np.arange(total_samples)
        
        logger.info(f"Splitting {total_samples} samples with strategy: {self.split_strategy}")
        
        # Skip test split if test_size is 0 or negative
        test_indices = np.array([], dtype=int)
        train_val_indices = indices
        
        if self.test_size > 0:
            # Handle test split first if needed
            if self.split_strategy == 'temporal' or self.split_strategy == 'hard_cutoff':
                # For temporal strategies, take the most recent data as test
                sorted_indices = indices[np.argsort(data['timestep'])]
                test_count = int(total_samples * self.test_size)
                train_val_indices = sorted_indices[:-test_count]
                test_indices = sorted_indices[-test_count:]
            else:
                # For other strategies, use random test split
                train_val_indices, test_indices = train_test_split(
                    indices, 
                    test_size=self.test_size,
                    random_state=self.random_seed
                )
        
        # Now split train_val into train and validation
        if self.split_strategy == 'random':
            train_indices, val_indices = self._random_split(train_val_indices, data)
        elif self.split_strategy == 'temporal':
            train_indices, val_indices = self._temporal_split(train_val_indices, data)
        elif self.split_strategy == 'stratified':
            train_indices, val_indices = self._stratified_split(train_val_indices, data)
        elif self.split_strategy == 'stratified_temporal':
            train_indices, val_indices = self._stratified_temporal_split(train_val_indices, data)
        elif self.split_strategy == 'shuffle_weeks':
            train_indices, val_indices = self._shuffle_weeks_split(train_val_indices, data)
        elif self.split_strategy == 'hard_cutoff':
            train_indices, val_indices = self._hard_cutoff_split(train_val_indices, data)
        else:
            logger.warning(f"Unknown split strategy: {self.split_strategy}. Using random split.")
            train_indices, val_indices = self._random_split(train_val_indices, data)
        
        # Apply time gap if specified
        if self.gap_days > 0 and 'timestep' in data:
            max_train_time = data['timestep'][train_indices].max()
            gap_threshold = max_train_time + np.timedelta64(self.gap_days, 'D')
            val_indices = val_indices[data['timestep'][val_indices] >= gap_threshold]
            logger.info(f"Applied {self.gap_days}-day gap between train and validation")
        
        # Compute and log statistics about the splits
        self._compute_split_statistics(data, train_indices, val_indices, test_indices)
        self._log_split_summary()
        
        # Generate visualizations
        self._generate_split_visualizations(data, train_indices, val_indices, test_indices)
        
        return train_indices, val_indices, test_indices
    
    def _random_split(
        self, 
        indices: np.ndarray, 
        data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform random splitting.
        
        Args:
            indices: Array of indices to split
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=self.val_size,
            random_state=self.random_seed
        )
        
        return train_indices, val_indices
    
    def _temporal_split(
        self, 
        indices: np.ndarray, 
        data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform temporal splitting based on timesteps.
        
        Args:
            indices: Array of indices to split
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        if 'timestep' not in data:
            logger.warning("No timestep information found. Falling back to random split.")
            return self._random_split(indices, data)
        
        # Get timesteps and sort indices
        timesteps = data['timestep']
        sorted_indices = indices[np.argsort(timesteps[indices])]
        
        # Determine split point
        train_count = int(len(sorted_indices) * (1 - self.val_size))
        
        # Split indices
        train_indices = sorted_indices[:train_count]
        val_indices = sorted_indices[train_count:]
        
        return train_indices, val_indices
    
    def _stratified_split(
        self, 
        indices: np.ndarray, 
        data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform stratified splitting.
        
        Args:
            indices: Array of indices to split
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        # Define a stratification target
        strat_target = None
        
        if self.stratify_key is not None and self.stratify_key in data:
            # Use specified stratification key
            strat_values = data[self.stratify_key][indices]
            if strat_values.ndim > 1:
                # Handle multi-dimensional stratification keys
                # E.g., use the first column or create a hash
                strat_target = np.array([hash(tuple(v)) for v in strat_values])
            else:
                strat_target = strat_values
        else:
            # Try to create a stratification target from rho values
            if 'act_rho' in data:
                # Get mean rho value for each sample
                strat_target = []
                for i in indices:
                    rho_data = data['act_rho'][i]
                    if hasattr(rho_data, 'dtype') and rho_data.dtype.names and 'rho_max' in rho_data.dtype.names:
                        rho_values = rho_data['rho_max']
                    else:
                        rho_values = np.array([item[1] if isinstance(item, tuple) else item for item in rho_data])
                    
                    strat_target.append(np.mean(rho_values))
                
                strat_target = np.array(strat_target)
                
                # Bin rho values for stratification
                n_bins = 10
                bins = np.linspace(np.min(strat_target), np.max(strat_target), n_bins + 1)
                strat_target = np.digitize(strat_target, bins)
        
        # Fall back to random if no stratification target
        if strat_target is None:
            logger.warning("No stratification target found. Falling back to random split.")
            return self._random_split(indices, data)
        
        # Perform stratified split
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=self.val_size,
            random_state=self.random_seed,
            stratify=strat_target
        )
        
        return train_indices, val_indices
    
    def _stratified_temporal_split(
        self, 
        indices: np.ndarray, 
        data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform stratified temporal splitting (time-aware per action).
        Takes the most recent samples for each action category for validation.
        
        Args:
            indices: Array of indices to split
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        if 'timestep' not in data:
            logger.warning("No timestep information found. Falling back to stratified split.")
            return self._stratified_split(indices, data)
        
        # Get action categories from rho data or soft labels
        action_categories = self._get_action_categories(indices, data)
        
        if action_categories is None:
            logger.warning("Could not determine action categories. Falling back to temporal split.")
            return self._temporal_split(indices, data)
        
        # Perform temporal split per action category
        val_indices = []
        for category in np.unique(action_categories):
            # Get indices for this category
            category_indices = indices[action_categories[indices] == category]
            
            # Sort by time
            sorted_indices = category_indices[np.argsort(data['timestep'][category_indices])]
            
            # Take the most recent samples for validation
            val_count = int(len(sorted_indices) * self.val_size)
            val_indices.append(sorted_indices[-val_count:])
        
        # Combine validation indices
        val_indices = np.concatenate(val_indices)
        
        # Training indices are everything else
        train_indices = np.setdiff1d(indices, val_indices)
        
        return train_indices, val_indices
    
    def _shuffle_weeks_split(
        self, 
        indices: np.ndarray, 
        data: Dict[str, np.ndarray],
        max_size: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform week-based stratified split, where entire weeks are assigned
        to either training or validation.
        
        Args:
            indices: Array of indices to split
            data: Data dictionary
            max_size: Maximum validation size as fraction of total
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        if 'timestep' not in data:
            logger.warning("No timestep information found. Falling back to random split.")
            return self._random_split(indices, data)
        
        # Get action categories
        action_categories = self._get_action_categories(indices, data)
        
        if action_categories is None:
            # If no action categories, just use weeks
            return self._shuffle_weeks_no_category(indices, data)
        
        # Perform week-based split per action category
        val_indices = []
        
        for category in np.unique(action_categories):
            # Get indices for this category
            category_indices = indices[action_categories[indices] == category]
            
            # Group by week
            timesteps = data['timestep'][category_indices]
            weeks = np.array([np.datetime64(ts, 'W') for ts in timesteps])
            
            # Count samples per week
            unique_weeks, week_counts = np.unique(weeks, return_counts=True)
            week_info = list(zip(unique_weeks, week_counts))
            
            # Shuffle weeks
            self.rng.shuffle(week_info)
            
            # Select weeks for validation until we reach desired size
            val_weeks = []
            val_count = 0
            target_count = int(len(category_indices) * self.val_size)
            
            for week, count in week_info:
                if val_count + count <= target_count:
                    val_weeks.append(week)
                    val_count += count
                elif val_count + count <= len(category_indices) * max_size:  # Check max_size
                    val_weeks.append(week)
                    val_count += count
                    break
            
            # Get indices for selected weeks
            cat_val_indices = []
            for i, week in zip(category_indices, weeks):
                if week in val_weeks:
                    cat_val_indices.append(i)
            
            # Fallback if not enough data selected
            if len(cat_val_indices) == 0:
                cat_val_indices = self.rng.choice(
                    category_indices, 
                    size=max(1, int(len(category_indices) * self.val_size)),
                    replace=False
                )
            
            val_indices.extend(cat_val_indices)
        
        # Convert to numpy array
        val_indices = np.array(val_indices)
        
        # Training indices are everything else
        train_indices = np.setdiff1d(indices, val_indices)
        
        return train_indices, val_indices
    
    def _shuffle_weeks_no_category(
        self, 
        indices: np.ndarray, 
        data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform week-based split without action categories.
        
        Args:
            indices: Array of indices to split
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        # Group by week
        timesteps = data['timestep'][indices]
        weeks = np.array([np.datetime64(ts, 'W') for ts in timesteps])
        
        # Count samples per week
        unique_weeks, week_counts = np.unique(weeks, return_counts=True)
        week_info = list(zip(unique_weeks, week_counts))
        
        # Shuffle weeks
        self.rng.shuffle(week_info)
        
        # Select weeks for validation until we reach desired size
        val_weeks = []
        val_count = 0
        target_count = int(len(indices) * self.val_size)
        
        for week, count in week_info:
            if val_count + count <= target_count:
                val_weeks.append(week)
                val_count += count
            elif val_count == 0:  # Ensure at least one week is selected
                val_weeks.append(week)
                val_count += count
                break
        
        # Get indices for selected weeks
        val_indices = []
        for i, week in zip(indices, weeks):
            if week in val_weeks:
                val_indices.append(i)
        
        val_indices = np.array(val_indices)
        
        # Training indices are everything else
        train_indices = np.setdiff1d(indices, val_indices)
        
        return train_indices, val_indices
    
    def _hard_cutoff_split(
        self, 
        indices: np.ndarray, 
        data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hard cutoff split based on time.
        Similar to temporal split but with more descriptive name.
        
        Args:
            indices: Array of indices to split
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        # This is essentially the same as temporal split
        return self._temporal_split(indices, data)
    
    def _get_action_categories(
        self, 
        indices: np.ndarray, 
        data: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Extract action categories from the data for stratification.
        
        Args:
            indices: Array of indices to process
            data: Data dictionary
            
        Returns:
            Array of action categories or None if not available
        """
        action_categories = None
        
        # Try different ways to get action categories
        if 'best_action' in data:
            # Use best action if available
            action_categories = data['best_action']
        elif 'act_rho' in data:
            # Use the first action in act_rho (not ideal but a fallback)
            categories = []
            for i in indices:
                act_data = data['act_rho'][i]
                if hasattr(act_data, 'dtype') and act_data.dtype.names and 'action' in act_data.dtype.names:
                    categories.append(act_data['action'][0])
                elif isinstance(act_data[0], tuple):
                    categories.append(act_data[0][0])
                else:
                    return None
            action_categories = np.array(categories)
        
        if action_categories is not None:
            # Encode string categories to integers
            label_encoder = LabelEncoder()
            return label_encoder.fit_transform(action_categories)
        
        return None
    
    def _compute_split_statistics(
        self, 
        data: Dict[str, np.ndarray], 
        train_indices: np.ndarray, 
        val_indices: np.ndarray, 
        test_indices: np.ndarray
    ) -> None:
        """
        Compute statistics about the data splits.
        
        Args:
            data: Data dictionary
            train_indices: Training indices
            val_indices: Validation indices
            test_indices: Test indices
        """
        # Store basic stats
        self.split_stats['train']['count'] = len(train_indices)
        self.split_stats['val']['count'] = len(val_indices)
        self.split_stats['test']['count'] = len(test_indices)
        
        # Compute timestep statistics if available
        if 'timestep' in data:
            timesteps = data['timestep']
            
            if len(train_indices) > 0:
                train_times = timesteps[train_indices]
                self.split_stats['train']['time_min'] = np.min(train_times)
                self.split_stats['train']['time_max'] = np.max(train_times)
            
            if len(val_indices) > 0:
                val_times = timesteps[val_indices]
                self.split_stats['val']['time_min'] = np.min(val_times)
                self.split_stats['val']['time_max'] = np.max(val_times)
                
                # Check for future validation samples
                if len(train_indices) > 0:
                    future_val_mask = val_times > np.max(train_times)
                    future_val_count = np.sum(future_val_mask)
                    
                    if future_val_count > 0:
                        self.split_stats['val']['future_count'] = int(future_val_count)
                        self.split_stats['val']['future_percent'] = future_val_count / len(val_indices)
                        self.split_stats['val']['future_time_min'] = np.min(val_times[future_val_mask])
                        self.split_stats['val']['future_time_max'] = np.max(val_times[future_val_mask])
            
            if len(test_indices) > 0:
                test_times = timesteps[test_indices]
                self.split_stats['test']['time_min'] = np.min(test_times)
                self.split_stats['test']['time_max'] = np.max(test_times)
        
        # Compute label distribution statistics
        action_categories = self._get_action_categories(np.concatenate([train_indices, val_indices]), data)
        
        if action_categories is not None:
            # Compute action distribution metrics
            self._compute_action_distribution_metrics(action_categories, train_indices, val_indices)
        
        # Compute soft label drift metrics
        if 'soft_labels' in data:
            self._compute_soft_label_drift_metrics(data, train_indices, val_indices)
    
    def _compute_action_distribution_metrics(
        self, 
        action_categories: np.ndarray, 
        train_indices: np.ndarray, 
        val_indices: np.ndarray
    ) -> None:
        """
        Compute metrics about action category distributions.
        
        Args:
            action_categories: Array of action categories
            train_indices: Training indices
            val_indices: Validation indices
        """
        # Get counts for each action in train and val
        train_cats = action_categories[train_indices]
        val_cats = action_categories[val_indices]
        
        unique_cats = np.unique(np.concatenate([train_cats, val_cats]))
        
        # Count occurrences
        train_counts = np.zeros(len(unique_cats))
        val_counts = np.zeros(len(unique_cats))
        
        for i, cat in enumerate(unique_cats):
            train_counts[i] = np.sum(train_cats == cat)
            val_counts[i] = np.sum(val_cats == cat)
        
        # Normalize to get distributions
        train_dist = train_counts / np.sum(train_counts) if np.sum(train_counts) > 0 else train_counts
        val_dist = val_counts / np.sum(val_counts) if np.sum(val_counts) > 0 else val_counts
        
        # Compute Jensen-Shannon divergence
        jsd = jensenshannon(train_dist, val_dist)
        self.drift_metrics['action_jsd'] = float(jsd)
        
        # Compute unseen action rate
        train_unique = set(train_cats)
        val_unique = set(val_cats)
        unseen = val_unique - train_unique
        
        if val_unique:
            unseen_rate = len(unseen) / len(val_unique)
            self.drift_metrics['unseen_action_rate'] = float(unseen_rate)
    
    def _compute_soft_label_drift_metrics(
        self, 
        data: Dict[str, np.ndarray], 
        train_indices: np.ndarray, 
        val_indices: np.ndarray
    ) -> None:
        """
        Compute drift metrics for soft labels.
        
        Args:
            data: Data dictionary
            train_indices: Training indices
            val_indices: Validation indices
        """
        # Skip if not enough data
        if len(train_indices) == 0 or len(val_indices) == 0:
            return
        
        # Calculate label distributions
        train_dist = self._compute_label_distribution(data, train_indices)
        val_dist = self._compute_label_distribution(data, val_indices)
        
        # Calculate Jensen-Shannon divergence between distributions
        if train_dist is not None and val_dist is not None:
            # Ensure distributions have the same keys
            all_keys = set(train_dist.keys()) | set(val_dist.keys())
            
            # Create arrays with zeros for missing keys
            train_array = np.array([train_dist.get(k, 0) for k in all_keys])
            val_array = np.array([val_dist.get(k, 0) for k in all_keys])
            
            # Normalize
            train_array = train_array / np.sum(train_array)
            val_array = val_array / np.sum(val_array)
            
            # Calculate JSD
            jsd = jensenshannon(train_array, val_array)
            self.drift_metrics['label_jsd'] = float(jsd)
            
            # Calculate percentage of unseen labels in validation
            train_keys = set(train_dist.keys())
            val_keys = set(val_dist.keys())
            
            unseen_keys = val_keys - train_keys
            if val_keys:
                unseen_ratio = len(unseen_keys) / len(val_keys)
                self.drift_metrics['unseen_label_ratio'] = float(unseen_ratio)
        
        # Compute entropy statistics for soft labels
        if 'soft_labels' in data and 'timestep' in data:
            self._compute_temporal_entropy_stats(data, train_indices, val_indices)
    
    def _compute_label_distribution(self, data: Dict[str, np.ndarray], indices: np.ndarray) -> Optional[Dict[str, int]]:
        """
        Compute distribution of labels in a dataset split.
        
        Args:
            data: Data dictionary
            indices: Indices of the split
            
        Returns:
            Dictionary mapping label keys to counts, or None if no indices
        """
        if len(indices) == 0 or 'soft_labels' not in data:
            return None
        
        label_counts = {}
        
        # Count occurrences of each action key
        for idx in indices:
            soft_data = data['soft_labels'][idx]
            
            for item in soft_data:
                if isinstance(item, tuple):
                    action_key = str(item[0])
                elif hasattr(item, 'dtype') and item.dtype.names and 'action' in item.dtype.names:
                    action_key = str(item['action'])
                else:
                    continue
                
                label_counts[action_key] = label_counts.get(action_key, 0) + 1
        
        return label_counts
    
    def _compute_temporal_entropy_stats(
        self, 
        data: Dict[str, np.ndarray], 
        train_indices: np.ndarray, 
        val_indices: np.ndarray
    ) -> None:
        """
        Compute entropy statistics over time.
        
        Args:
            data: Data dictionary
            train_indices: Training indices
            val_indices: Validation indices
        """
        if len(train_indices) == 0 or 'timestep' not in data or 'soft_labels' not in data:
            return
        
        # Convert timesteps to datetime if needed
        timesteps = data['timestep']
        
        # Compute entropy for each sample
        entropy_values = []
        entropy_times = []
        split_labels = []  # 0 for train, 1 for val
        
        # Process training set
        for idx in train_indices:
            entropy = self._compute_soft_label_entropy(data['soft_labels'][idx])
            if entropy is not None:
                entropy_values.append(entropy)
                entropy_times.append(timesteps[idx])
                split_labels.append(0)  # 0 for train
        
        # Process validation set
        for idx in val_indices:
            entropy = self._compute_soft_label_entropy(data['soft_labels'][idx])
            if entropy is not None:
                entropy_values.append(entropy)
                entropy_times.append(timesteps[idx])
                split_labels.append(1)  # 1 for val
        
        # Store for visualization
        self.entropy_data = {
            'entropy': np.array(entropy_values),
            'time': np.array(entropy_times),
            'split': np.array(split_labels)
        }
        
        # Compute weekly JSD from training to validation
        if len(train_indices) > 0 and len(val_indices) > 0:
            self._compute_weekly_jsd(data, train_indices, val_indices)
    
    def _compute_soft_label_entropy(self, soft_labels) -> Optional[float]:
        """
        Compute entropy of soft labels.
        
        Args:
            soft_labels: Soft labels for a sample
            
        Returns:
            Entropy value or None if computation fails
        """
        try:
            # Extract soft scores
            soft_scores = []
            
            if hasattr(soft_labels, 'dtype') and soft_labels.dtype.names and 'soft_score' in soft_labels.dtype.names:
                soft_scores = soft_labels['soft_score']
            elif all(isinstance(item, tuple) for item in soft_labels):
                soft_scores = [item[1] for item in soft_labels]
            else:
                # Try direct approach
                soft_scores = soft_labels
            
            # Convert to numpy array and normalize
            soft_scores = np.array(soft_scores, dtype=float)
            if np.sum(soft_scores) > 0:
                soft_scores = soft_scores / np.sum(soft_scores)
            
            # Compute entropy
            entropy = -np.sum(soft_scores * np.log2(soft_scores + 1e-10))
            return float(entropy)
        except Exception as e:
            logger.warning(f"Error computing entropy: {e}")
            return None
    
    def _compute_weekly_jsd(
            self, 
            data: Dict[str, np.ndarray], 
            train_indices: np.ndarray, 
            val_indices: np.ndarray
        ) -> None:
            """
            Compute Jensen-Shannon divergence for each week in validation
            compared to the overall training distribution.
            
            Args:
                data: Data dictionary
                train_indices: Training indices
                val_indices: Validation indices
            """
            # Extract all soft labels distributions
            train_dists = []
            for idx in train_indices:
                if 'soft_labels' in data:
                    dist = self._extract_soft_label_distribution(data['soft_labels'][idx])
                    if dist is not None:
                        train_dists.append(dist)
            
            if not train_dists:
                return
            
            # Compute average distribution for training
            train_avg_dist = {}
            for dist in train_dists:
                for key, value in dist.items():
                    if key not in train_avg_dist:
                        train_avg_dist[key] = []
                    train_avg_dist[key].append(value)
            
            # Calculate mean for each key
            for key in train_avg_dist:
                train_avg_dist[key] = np.mean(train_avg_dist[key])
            
            # Normalize the average distribution
            total = sum(train_avg_dist.values())
            if total > 0:
                for key in train_avg_dist:
                    train_avg_dist[key] /= total
            
            # Group validation samples by week
            val_weeks = {}
            for idx in val_indices:
                if 'timestep' not in data or 'soft_labels' not in data:
                    continue
                    
                timestamp = data['timestep'][idx]
                week = np.datetime64(timestamp, 'W')
                
                if week not in val_weeks:
                    val_weeks[week] = []
                
                dist = self._extract_soft_label_distribution(data['soft_labels'][idx])
                if dist is not None:
                    val_weeks[week].append(dist)
            
            # Compute JSD for each week
            weekly_jsd = []
            for week, dists in val_weeks.items():
                if not dists:
                    continue
                    
                # Combine distributions for this week
                week_dist = {}
                for dist in dists:
                    for key, value in dist.items():
                        if key not in week_dist:
                            week_dist[key] = []
                        week_dist[key].append(value)
                
                # Calculate mean for each key
                for key in week_dist:
                    week_dist[key] = np.mean(week_dist[key])
                
                # Normalize the weekly distribution
                total = sum(week_dist.values())
                if total > 0:
                    for key in week_dist:
                        week_dist[key] /= total
                
                # Calculate JSD
                all_keys = set(train_avg_dist.keys()) | set(week_dist.keys())
                train_array = np.array([train_avg_dist.get(k, 0) for k in all_keys])
                week_array = np.array([week_dist.get(k, 0) for k in all_keys])
                
                # Normalize
                train_array = train_array / np.sum(train_array)
                week_array = week_array / np.sum(week_array)
                
                jsd = jensenshannon(train_array, week_array)
                weekly_jsd.append((week, float(jsd)))
            
            # Store for visualization
            self.weekly_jsd_data = sorted(weekly_jsd)
            
            # Store highest JSD week
            if weekly_jsd:
                max_jsd_week, max_jsd = max(weekly_jsd, key=lambda x: x[1])
                self.drift_metrics['max_weekly_jsd'] = max_jsd
                self.drift_metrics['max_jsd_week'] = str(max_jsd_week)

    def _extract_soft_label_distribution(self, soft_labels) -> Optional[Dict[str, float]]:
        """
        Extract probability distribution from soft labels.
        
        Args:
            soft_labels: Soft labels for a sample
            
        Returns:
            Dictionary mapping action keys to probabilities, or None if extraction fails
        """
        try:
            # Initialize distribution
            distribution = {}
            
            # Handle different formats of soft_labels
            if hasattr(soft_labels, 'dtype') and soft_labels.dtype.names:
                # Structured array
                if 'action' in soft_labels.dtype.names and 'soft_score' in soft_labels.dtype.names:
                    for i in range(len(soft_labels)):
                        key = str(soft_labels['action'][i])
                        value = float(soft_labels['soft_score'][i])
                        distribution[key] = value
            elif all(isinstance(item, tuple) for item in soft_labels):
                # List of tuples (action, score)
                for key, value in soft_labels:
                    distribution[str(key)] = float(value)
            
            # Normalize if needed
            total = sum(distribution.values())
            if total > 0:
                for key in distribution:
                    distribution[key] /= total
            
            return distribution if distribution else None
        except Exception as e:
            logger.warning(f"Error extracting soft label distribution: {e}")
            return None
    
    def _log_split_summary(self) -> None:
        """Log a summary of the split statistics."""
        train_count = self.split_stats['train'].get('count', 0)
        val_count = self.split_stats['val'].get('count', 0)
        test_count = self.split_stats['test'].get('count', 0)
        
        summary = [
            "=== Split Summary ===",
            f"Train samples: {train_count}",
            f"Val samples:   {val_count}",
            f"Test samples:  {test_count}"
        ]
        
        # Add time range info if available
        if 'time_min' in self.split_stats['train'] and 'time_max' in self.split_stats['train']:
            summary.append(f"Train time: {self.split_stats['train']['time_min']} → {self.split_stats['train']['time_max']}")
        
        if 'time_min' in self.split_stats['val'] and 'time_max' in self.split_stats['val']:
            summary.append(f"Val time:   {self.split_stats['val']['time_min']} → {self.split_stats['val']['time_max']}")
        
        if 'time_min' in self.split_stats['test'] and 'time_max' in self.split_stats['test']:
            summary.append(f"Test time:  {self.split_stats['test']['time_min']} → {self.split_stats['test']['time_max']}")
        
        # Add drift metrics
        if 'label_jsd' in self.drift_metrics:
            summary.append(f"JSD Drift (train vs val label dist): {self.drift_metrics['label_jsd']:.4f}")
        
        if 'action_jsd' in self.drift_metrics:
            summary.append(f"JSD Drift (train vs val action dist): {self.drift_metrics['action_jsd']:.4f}")
        
        if 'unseen_label_ratio' in self.drift_metrics:
            summary.append(f"Rate of unseen labels in val:       {self.drift_metrics['unseen_label_ratio']:.2%}")
        
        if 'max_weekly_jsd' in self.drift_metrics:
            summary.append(f"Maximum weekly JSD: {self.drift_metrics['max_weekly_jsd']:.4f} (week {self.drift_metrics['max_jsd_week']})")
        
        # Add warning for future validation samples
        if 'future_percent' in self.split_stats['val'] and self.split_stats['val']['future_percent'] > 0:
            future_pct = self.split_stats['val']['future_percent']
            summary.append(f"⚠️  {future_pct:.2%} of val samples are from future relative to train")
            
            # Add future time range
            if 'future_time_min' in self.split_stats['val'] and 'future_time_max' in self.split_stats['val']:
                future_min = self.split_stats['val']['future_time_min']
                future_max = self.split_stats['val']['future_time_max']
                summary.append(f"   Future val time range: {future_min} → {future_max}")
        else:
            summary.append("✅ No val sample comes after max(train time)")
        
        # Add note about external test data if needed
        if self.test_size <= 0 and self.split_stats['test']['count'] == 0:
            summary.append("Note: No test data in this split (use external test file if needed)")
        
        # Log the summary
        for line in summary:
            logger.info(line)
    
    def _generate_split_visualizations(
        self, 
        data: Dict[str, np.ndarray], 
        train_indices: np.ndarray, 
        val_indices: np.ndarray, 
        test_indices: np.ndarray
    ) -> None:
        """
        Generate visualizations of the splits.
        
        Args:
            data: Data dictionary
            train_indices: Training indices
            val_indices: Validation indices
            test_indices: Test indices
        """
        # Ensure visualization directory exists
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Create split summary visualization
        self._plot_split_summary()
        
        # Create temporal visualizations if data is available
        if hasattr(self, 'entropy_data'):
            self._plot_temporal_entropy()
        
        # Create weekly JSD visualization if available
        if hasattr(self, 'weekly_jsd_data'):
            self._plot_temporal_jsd()
            
        # Create action distribution visualization
        self._plot_action_distribution(data, train_indices, val_indices, test_indices)
    
    def _plot_split_summary(self) -> None:
        """Create and save a summary visualization of the splits."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot sample counts
        train_count = self.split_stats['train'].get('count', 0)
        val_count = self.split_stats['val'].get('count', 0)
        test_count = self.split_stats['test'].get('count', 0)
        
        splits = ['Train', 'Validation', 'Test']
        counts = [train_count, val_count, test_count]
        
        ax1.bar(splits, counts, color=['#2c7fb8', '#7fcdbb', '#edf8b1'])
        ax1.set_title('Sample Counts by Split')
        ax1.set_ylabel('Number of Samples')
        
        for i, count in enumerate(counts):
            ax1.text(i, count + 0.1, str(count), ha='center')
        
        # Plot drift metrics if available
        metrics = []
        values = []
        
        if 'label_jsd' in self.drift_metrics:
            metrics.append('JSD (Labels)')
            values.append(self.drift_metrics['label_jsd'])
            
        if 'action_jsd' in self.drift_metrics:
            metrics.append('JSD (Actions)')
            values.append(self.drift_metrics['action_jsd'])
            
        if 'unseen_label_ratio' in self.drift_metrics:
            metrics.append('Unseen Label Rate')
            values.append(self.drift_metrics['unseen_label_ratio'])
            
        if 'max_weekly_jsd' in self.drift_metrics:
            metrics.append('Max Weekly JSD')
            values.append(self.drift_metrics['max_weekly_jsd'])
        
        if metrics:
            ax2.bar(metrics, values, color=['#fc8d59', '#d73027', '#fee090', '#91bfdb'])
            ax2.set_title('Drift Metrics')
            
            for i, value in enumerate(values):
                ax2.text(i, value + 0.01, f"{value:.4f}", ha='center')
        else:
            ax2.text(0.5, 0.5, "No drift metrics available", ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'split_summary.png'))
        plt.close()
    
    def _plot_temporal_entropy(self) -> None:
        """Create and save a visualization of entropy over time."""
        if not hasattr(self, 'entropy_data'):
            return
        
        # Convert data to pandas DataFrame for easier plotting
        df = pd.DataFrame({
            'time': self.entropy_data['time'],
            'entropy': self.entropy_data['entropy'],
            'split': [('Train' if s == 0 else 'Validation') for s in self.entropy_data['split']]
        })
        
        # Convert time to pandas datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Add week column
        df['year'] = df['time'].dt.isocalendar().year
        df['week'] = df['time'].dt.isocalendar().week
        df['yearweek'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)
        
        # Calculate weekly entropy means
        weekly_means = df.groupby(['yearweek', 'split'])['entropy'].mean().reset_index()
        
        # Sort by yearweek for proper ordering
        weekly_means['year_num'] = weekly_means['yearweek'].str.split('-').str[0].astype(int)
        weekly_means['week_num'] = weekly_means['yearweek'].str.split('-').str[1].astype(int)
        weekly_means = weekly_means.sort_values(['year_num', 'week_num'])
        
        # Plotting
        plt.figure(figsize=(14, 6))
        
        # Use seaborn for better colors
        sns.barplot(x='yearweek', y='entropy', hue='split', data=weekly_means, palette='Set1')
        
        plt.title('Mean Soft Label Entropy by Week')
        plt.xlabel('Year-Week')
        plt.ylabel('Entropy')
        plt.xticks(rotation=90)
        plt.legend(title='Split')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Optimize x-axis labels for readability
        ax = plt.gca()
        if len(weekly_means['yearweek'].unique()) > 20:
            # Show only every Nth label to avoid overcrowding
            n = max(1, len(weekly_means['yearweek'].unique()) // 20)
            for i, label in enumerate(ax.get_xticklabels()):
                if i % n != 0:
                    label.set_visible(False)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.visualization_dir, 'weekly_entropy.png'))
        plt.close()
    
    def _plot_temporal_jsd(self) -> None:
        """Create and save a visualization of Jensen-Shannon divergence over time."""
        if not hasattr(self, 'weekly_jsd_data') or not self.weekly_jsd_data:
            return
        
        # Convert to pandas DataFrame
        weeks, jsds = zip(*self.weekly_jsd_data)
        
        # Convert weeks to strings for plotting
        week_strs = [str(w).split('T')[0] for w in weeks]
        
        # Plotting
        plt.figure(figsize=(14, 6))
        
        # Create the bar plot
        plt.bar(week_strs, jsds, color='#4292c6')
        
        plt.title('Weekly JSD from Training Distribution')
        plt.xlabel('Year-Week')
        plt.ylabel('Jensen-Shannon Divergence')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Optimize x-axis labels for readability
        ax = plt.gca()
        if len(week_strs) > 20:
            # Show only every Nth label to avoid overcrowding
            n = max(1, len(week_strs) // 20)
            for i, label in enumerate(ax.get_xticklabels()):
                if i % n != 0:
                    label.set_visible(False)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.visualization_dir, 'weekly_jsd.png'))
        plt.close()
    
    def _plot_action_distribution(
        self, 
        data: Dict[str, np.ndarray], 
        train_indices: np.ndarray, 
        val_indices: np.ndarray, 
        test_indices: np.ndarray
    ) -> None:
        """
        Create and save a visualization of action distribution.
        
        Args:
            data: Data dictionary
            train_indices: Training indices
            val_indices: Validation indices
            test_indices: Test indices
        """
        # Get action categories
        action_categories = self._get_action_categories(
            np.concatenate([train_indices, val_indices, test_indices]), 
            data
        )
        
        if action_categories is None:
            return
        
        # Create a DataFrame for easier plotting
        all_indices = np.concatenate([train_indices, val_indices, test_indices])
        split_labels = np.zeros(len(all_indices), dtype=int)
        split_labels[len(train_indices):len(train_indices) + len(val_indices)] = 1
        split_labels[len(train_indices) + len(val_indices):] = 2
        
        df = pd.DataFrame({
            'index': all_indices,
            'action': action_categories,
            'split': [['Train', 'Validation', 'Test'][s] for s in split_labels]
        })
        
        # Count actions per split
        action_counts = df.groupby(['split', 'action']).size().reset_index(name='count')
        
        # Replace numeric action categories with string labels for better readability
        unique_actions = sorted(df['action'].unique())
        action_counts['action'] = action_counts['action'].astype(str)
        
        # Plotting
        plt.figure(figsize=(14, 8))
        
        # Use seaborn for better visualization
        g = sns.catplot(
            x='action', 
            y='count', 
            hue='split', 
            data=action_counts, 
            kind='bar', 
            palette='Set2',
            height=6, 
            aspect=1.5
        )
        
        plt.title('Action Distribution by Split')
        plt.xlabel('Action Category')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Optimize x-axis labels for readability
        ax = plt.gca()
        if len(unique_actions) > 20:
            # Show only every Nth label to avoid overcrowding
            n = max(1, len(unique_actions) // 20)
            for i, label in enumerate(ax.get_xticklabels()):
                if i % n != 0:
                    label.set_visible(False)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.visualization_dir, 'action_distribution.png'))
        plt.close()
    
    def save_report(self, filename: str = 'split_report.json') -> None:
        """
        Save a detailed report of the split statistics.
        
        Args:
            filename: Name of the report file
        """
        report = {
            'config': {
                'split_strategy': self.split_strategy,
                'val_size': self.val_size,
                'test_size': self.test_size,
                'random_seed': self.random_seed,
                'gap_days': self.gap_days
            },
            'split_stats': self.split_stats,
            'drift_metrics': self.drift_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save as JSON
        with open(os.path.join(self.visualization_dir, filename), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved split report to {os.path.join(self.visualization_dir, filename)}")