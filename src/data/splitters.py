"""
Advanced data splitting strategies for model evaluation.
Provides various techniques for creating robust train/validation/test splits.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit
from collections import Counter
import datetime

# Configure logging
logger = logging.getLogger(__name__)

class SplitStrategy:
    """Base class for all splitting strategies."""
    
    def __init__(self, val_size: float = 0.2, random_seed: int = 42):
        """
        Initialize the split strategy.
        
        Args:
            val_size: Validation set size as a fraction
            random_seed: Random seed for reproducibility
        """
        self.val_size = val_size
        self.random_seed = random_seed
    
    def split(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        raise NotImplementedError("Subclasses must implement split")


class RandomSplit(SplitStrategy):
    """Random splitting strategy with optional stratification."""
    
    def __init__(
        self, 
        val_size: float = 0.2, 
        random_seed: int = 42, 
        stratify_key: Optional[str] = None,
        stratify_fn: Optional[Callable] = None
    ):
        """
        Initialize the random split strategy.
        
        Args:
            val_size: Validation set size as a fraction
            random_seed: Random seed for reproducibility
            stratify_key: Key in data to use for stratification
            stratify_fn: Function to compute stratification values
        """
        super().__init__(val_size, random_seed)
        self.stratify_key = stratify_key
        self.stratify_fn = stratify_fn
    
    def split(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data randomly into training and validation sets.
        
        Args:
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        if 'obs' not in data:
            raise ValueError("Data must contain 'obs' key")
        
        num_samples = len(data['obs'])
        indices = np.arange(num_samples)
        
        # Determine stratification
        stratify = None
        if self.stratify_key is not None and self.stratify_key in data:
            stratify = data[self.stratify_key]
        elif self.stratify_fn is not None:
            stratify = self.stratify_fn(data)
        
        # Apply stratification if possible
        if stratify is not None:
            # Check if stratify has the right shape
            if len(stratify) != num_samples:
                logger.warning(f"Stratification values shape {stratify.shape} doesn't match data shape {num_samples}. Disabling stratification.")
                stratify = None
        
        # Perform the split
        train_indices, val_indices = train_test_split(
            indices,
            test_size=self.val_size,
            random_state=self.random_seed,
            stratify=stratify
        )
        
        logger.info(f"Random split: {len(train_indices)} training samples, {len(val_indices)} validation samples")
        if stratify is not None:
            logger.info("Used stratification for splitting")
        
        return train_indices, val_indices


class TemporalSplit(SplitStrategy):
    """Temporal splitting strategy based on timesteps."""
    
    def __init__(
        self, 
        val_size: float = 0.2, 
        random_seed: int = 42, 
        timestep_key: str = 'timestep',
        reverse: bool = False
    ):
        """
        Initialize the temporal split strategy.
        
        Args:
            val_size: Validation set size as a fraction
            random_seed: Random seed for reproducibility (not used in temporal splits)
            timestep_key: Key for timestep data
            reverse: If True, oldest data goes to validation
        """
        super().__init__(val_size, random_seed)
        self.timestep_key = timestep_key
        self.reverse = reverse
    
    def split(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data temporally into training and validation sets.
        
        Args:
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        if 'obs' not in data:
            raise ValueError("Data must contain 'obs' key")
        
        if self.timestep_key not in data:
            logger.warning(f"Timestep key '{self.timestep_key}' not found in data. Falling back to random split.")
            return RandomSplit(self.val_size, self.random_seed).split(data)
        
        # Get timesteps
        timesteps = data[self.timestep_key]
        num_samples = len(timesteps)
        
        # Sort indices by timestep
        sorted_indices = np.argsort(timesteps)
        
        # Reverse order if specified (oldest data for validation)
        if self.reverse:
            sorted_indices = sorted_indices[::-1]
        
        # Split with latest timesteps for validation
        train_size = int(num_samples * (1 - self.val_size))
        train_indices = sorted_indices[:train_size]
        val_indices = sorted_indices[train_size:]
        
        # Get actual time ranges for logging
        train_times = timesteps[train_indices]
        val_times = timesteps[val_indices]
        
        logger.info(f"Temporal split: {len(train_indices)} training samples, {len(val_indices)} validation samples")
        logger.info(f"Training time range: {min(train_times)} to {max(train_times)}")
        logger.info(f"Validation time range: {min(val_times)} to {max(val_times)}")
        
        return train_indices, val_indices


class PerformanceSplit(SplitStrategy):
    """
    Performance-based split that ensures validation set has representative 
    coverage of different performance levels.
    """
    
    def __init__(
        self, 
        val_size: float = 0.2, 
        random_seed: int = 42, 
        rho_key: str = 'act_rho',
        num_bins: int = 5
    ):
        """
        Initialize the performance split strategy.
        
        Args:
            val_size: Validation set size as a fraction
            random_seed: Random seed for reproducibility
            rho_key: Key for rho values
            num_bins: Number of bins for stratification
        """
        super().__init__(val_size, random_seed)
        self.rho_key = rho_key
        self.num_bins = num_bins
    
    def split(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data based on performance (rho values) into training and validation sets.
        
        Args:
            data: Data dictionary
            
        Returns:
            Tuple of (training indices, validation indices)
        """
        if 'obs' not in data:
            raise ValueError("Data must contain 'obs' key")
        
        if self.rho_key not in data:
            logger.warning(f"Rho key '{self.rho_key}' not found in data. Falling back to random split.")
            return RandomSplit(self.val_size, self.random_seed).split(data)
        
        # Calculate performance metric for each sample
        num_samples = len(data['obs'])
        performance_metrics = np.zeros(num_samples)
        
        # Extract rho values
        for i in range(num_samples):
            rho_data = data[self.rho_key][i]
            
            # Handle structured array format
            if hasattr(rho_data, 'dtype') and rho_data.dtype.names and 'rho_max' in rho_data.dtype.names:
                rho_values = rho_data['rho_max']
            else:
                # Handle tuple format
                rho_values = np.array([item[1] if isinstance(item, tuple) else item for item in rho_data])
            
            # Use mean rho as performance metric
            performance_metrics[i] = np.mean(rho_values)
        
        # Create bins for stratification
        bins = np.linspace(
            np.min(performance_metrics),
            np.max(performance_metrics),
            self.num_bins + 1
        )
        bin_indices = np.digitize(performance_metrics, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        # Log bin distribution
        bin_counts = Counter(bin_indices)
        logger.info(f"Performance bins distribution: {bin_counts}")
        
        # Perform stratified split
        indices = np.arange(num_samples)
        train_indices, val_indices = train_test_split(
            indices,
            test_size=self.val_size,
            random_state=self.random_seed,
            stratify=bin_indices
        )
        
        logger.info(f"Performance split: {len(train_indices)} training samples, {len(val_indices)} validation samples")
        
        return train_indices, val_indices


class CrossValidationSplit:
    """
    Cross-validation split generator for K-fold validation.
    """
    
    def __init__(
        self, 
        n_splits: int = 5, 
        random_seed: int = 42, 
        strategy: str = 'kfold',
        stratify_key: Optional[str] = None,
        temporal_key: Optional[str] = None
    ):
        """
        Initialize the cross-validation split generator.
        
        Args:
            n_splits: Number of folds
            random_seed: Random seed for reproducibility
            strategy: Splitting strategy ('kfold', 'stratified', 'temporal')
            stratify_key: Key to use for stratification (only for 'stratified')
            temporal_key: Key for temporal ordering (only for 'temporal')
        """
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.strategy = strategy
        self.stratify_key = stratify_key
        self.temporal_key = temporal_key
        
        # Initialize splitter based on strategy
        if strategy == 'kfold':
            self.splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        elif strategy == 'stratified':
            self.splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        elif strategy == 'temporal':
            self.splitter = TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown cross-validation strategy: {strategy}")
    
    def get_splits(self, data: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate cross-validation splits.
        
        Args:
            data: Data dictionary
            
        Returns:
            List of (train_indices, val_indices) tuples, one for each fold
        """
        if 'obs' not in data:
            raise ValueError("Data must contain 'obs' key")
        
        num_samples = len(data['obs'])
        indices = np.arange(num_samples)
        
        # Get split groups for stratified or temporal
        groups = None
        if self.strategy == 'stratified' and self.stratify_key is not None:
            if self.stratify_key not in data:
                logger.warning(f"Stratify key '{self.stratify_key}' not found. Falling back to simple k-fold.")
                self.splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            else:
                groups = data[self.stratify_key]
        elif self.strategy == 'temporal' and self.temporal_key is not None:
            if self.temporal_key not in data:
                logger.warning(f"Temporal key '{self.temporal_key}' not found. Falling back to simple k-fold.")
                self.splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        
        # Generate splits
        splits = []
        for train_idx, val_idx in self.splitter.split(indices, groups):
            splits.append((train_idx, val_idx))
        
        logger.info(f"Generated {len(splits)} cross-validation splits with strategy: {self.strategy}")
        for i, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Fold {i+1}: {len(train_idx)} training samples, {len(val_idx)} validation samples")
        
        return splits


# Factory function for creating split strategies
def create_split_strategy(config: Dict[str, Any]) -> SplitStrategy:
    """
    Create a split strategy based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Split strategy instance
    """
    strategy_type = config.get('split_strategy', 'random')
    val_size = config.get('val_size', 0.2)
    random_seed = config.get('random_seed', 42)
    
    if strategy_type == 'random':
        return RandomSplit(
            val_size=val_size,
            random_seed=random_seed,
            stratify_key=config.get('stratify_key')
        )
    elif strategy_type == 'temporal':
        return TemporalSplit(
            val_size=val_size,
            random_seed=random_seed,
            timestep_key=config.get('timestep_key', 'timestep'),
            reverse=config.get('reverse_temporal', False)
        )
    elif strategy_type == 'performance':
        return PerformanceSplit(
            val_size=val_size,
            random_seed=random_seed,
            rho_key=config.get('rho_key', 'act_rho'),
            num_bins=config.get('num_bins', 5)
        )
    else:
        logger.warning(f"Unknown split strategy: {strategy_type}. Using random split.")
        return RandomSplit(val_size=val_size, random_seed=random_seed)