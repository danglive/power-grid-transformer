"""
Base module for data handling in the Q-Transformer framework.
Contains core abstractions and interfaces for dataset handling.
"""

import abc
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import os
import json
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataStats:
    """Statistics about a dataset."""
    num_samples: int
    obs_shape: Tuple[int, ...]
    action_dim: int
    num_actions: int
    feature_mins: np.ndarray
    feature_maxs: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    rho_min: float
    rho_max: float
    rho_mean: float
    rho_std: float
    
    def log_summary(self):
        """Log a summary of the dataset statistics."""
        logger.info(f"Dataset Statistics:")
        logger.info(f"  Number of samples: {self.num_samples}")
        logger.info(f"  Observation shape: {self.obs_shape}")
        logger.info(f"  Action dimension: {self.action_dim}")
        logger.info(f"  Actions per sample: {self.num_actions}")
        logger.info(f"  Feature range: [{self.feature_mins.min():.4f}, {self.feature_maxs.max():.4f}]")
        logger.info(f"  Feature mean range: [{self.feature_means.min():.4f}, {self.feature_means.max():.4f}]")
        logger.info(f"  Rho range: [{self.rho_min:.4f}, {self.rho_max:.4f}]")
        logger.info(f"  Rho mean/std: {self.rho_mean:.4f}/{self.rho_std:.4f}")
    
    def save(self, path):
        """Save the statistics to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Load statistics from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class BaseDataModule(abc.ABC):
    """
    Abstract base class for all data modules in the Q-Transformer framework.
    Provides the interface for loading and processing datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data module with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cache_dir = Path(config.get('cache_dir', 'data_cache'))
        self.dataset_dir = Path(config.get('dataset_dir', './'))
        self.save_to_cache = config.get('save_to_cache', True)
        self.stats = None  # Will be computed during setup
        
        # Ensure cache directory exists
        if self.save_to_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    @abc.abstractmethod
    def setup(self):
        """
        Set up the data module, loading and processing datasets.
        This method should be implemented by subclasses.
        """
        pass
    
    @abc.abstractmethod
    def get_train_dataset(self) -> Dataset:
        """
        Get the training dataset.
        
        Returns:
            Training dataset
        """
        pass
    
    @abc.abstractmethod
    def get_val_dataset(self) -> Dataset:
        """
        Get the validation dataset.
        
        Returns:
            Validation dataset
        """
        pass
    
    @abc.abstractmethod
    def get_test_dataset(self) -> Dataset:
        """
        Get the test dataset.
        
        Returns:
            Test dataset
        """
        pass
    
    def compute_stats(self, datasets: Union[Dataset, List[Dataset]]) -> DataStats:
        """
        Compute statistics from the given datasets.
        
        Args:
            datasets: Dataset or list of datasets to compute statistics from
            
        Returns:
            Dataset statistics
        """
        if not isinstance(datasets, list):
            datasets = [datasets]
        
        logger.info("Computing dataset statistics...")
        start_time = time.time()
        
        # Collect statistics across all datasets
        all_obs = []
        all_rho_values = []
        num_samples = 0
        obs_shape = None
        action_dim = None
        num_actions = None
        
        # Sample a subset of data for efficiency
        max_samples_per_dataset = 10000
        
        for dataset in datasets:
            # Get dataset size and determine sampling
            ds_size = len(dataset)
            num_samples += ds_size
            sample_indices = np.random.choice(ds_size, min(max_samples_per_dataset, ds_size), replace=False)
            
            # Sample observations and rho values
            for idx in sample_indices:
                sample = dataset[idx]
                
                # Get observation shape
                if obs_shape is None:
                    obs_shape = sample['observation'].shape
                
                # Get action dimension
                if action_dim is None and 'action_vectors' in sample:
                    action_dim = sample['action_vectors'].shape[-1]
                
                # Get number of actions
                if num_actions is None:
                    num_actions = len(sample['rho_values'])
                
                # Collect observation
                all_obs.append(sample['observation'].numpy())
                
                # Collect rho values
                all_rho_values.extend(sample['rho_values'].numpy())
        
        # Convert to arrays
        all_obs = np.vstack(all_obs)
        all_rho_values = np.array(all_rho_values)
        
        # Compute statistics
        feature_mins = np.min(all_obs, axis=0)
        feature_maxs = np.max(all_obs, axis=0)
        feature_means = np.mean(all_obs, axis=0)
        feature_stds = np.std(all_obs, axis=0)
        
        rho_min = np.min(all_rho_values)
        rho_max = np.max(all_rho_values)
        rho_mean = np.mean(all_rho_values)
        rho_std = np.std(all_rho_values)
        
        # Create stats object
        stats = DataStats(
            num_samples=num_samples,
            obs_shape=obs_shape,
            action_dim=action_dim,
            num_actions=num_actions,
            feature_mins=feature_mins,
            feature_maxs=feature_maxs,
            feature_means=feature_means,
            feature_stds=feature_stds,
            rho_min=rho_min,
            rho_max=rho_max,
            rho_mean=rho_mean,
            rho_std=rho_std
        )
        
        logger.info(f"Statistics computation completed in {time.time() - start_time:.2f} seconds")
        stats.log_summary()
        
        # Cache statistics
        if self.save_to_cache:
            stats_file = self.cache_dir / "dataset_stats.pkl"
            stats.save(stats_file)
            logger.info(f"Statistics cached to {stats_file}")
        
        return stats
    
    def get_stats(self) -> DataStats:
        """
        Get dataset statistics.
        
        Returns:
            Dataset statistics
        """
        if self.stats is None:
            # Try to load from cache
            stats_file = self.cache_dir / "dataset_stats.pkl"
            if self.save_to_cache and stats_file.exists():
                logger.info(f"Loading statistics from cache: {stats_file}")
                self.stats = DataStats.load(stats_file)
                self.stats.log_summary()
            else:
                logger.info("Statistics not found in cache. Computing...")
                self.stats = self.compute_stats([
                    self.get_train_dataset(),
                    self.get_val_dataset()
                ])
        
        return self.stats


class BaseDataset(Dataset, abc.ABC):
    """
    Abstract base class for all datasets in the Q-Transformer framework.
    Provides common functionality for data loading and processing.
    """
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """
        Initialize the dataset.
        
        Args:
            config: Configuration dictionary
            is_training: Whether this dataset is for training
        """
        self.config = config
        self.is_training = is_training
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Vector scaling flag (for feature normalization)
        self.vector_scaler = config.get('vector_scaler', False)
        
        # Load statistics if available (for scaling)
        self.stats = None
        if self.vector_scaler:
            stats_file = Path(config.get('cache_dir', 'data_cache')) / "dataset_stats.pkl"
            if stats_file.exists():
                self.stats = DataStats.load(stats_file)
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        pass
    
    def normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Normalize observation using dataset statistics.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Normalized observation tensor
        """
        if not self.vector_scaler or self.stats is None:
            return obs
        
        # Convert to numpy if tensor
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
            # Normalize using means and stds
            epsilon = 1e-7  # To avoid division by zero
            normalized = (obs_np - self.stats.feature_means) / (self.stats.feature_stds + epsilon)
            # Convert back to tensor
            return torch.FloatTensor(normalized)
        else:
            # Normalize numpy array
            epsilon = 1e-7
            normalized = (obs - self.stats.feature_means) / (self.stats.feature_stds + epsilon)
            return normalized
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Default collate function for batching samples.
        
        Args:
            batch: List of samples to collate
            
        Returns:
            Batched dictionary
        """
        # Check if batch is empty
        if not batch:
            return {}
        
        # Initialize result dictionary
        result = {}
        
        # Get all keys from the first sample
        keys = batch[0].keys()
        
        # Process each key
        for key in keys:
            # Skip None values
            if batch[0][key] is None:
                result[key] = None
                continue
            
            # Process based on value type
            if isinstance(batch[0][key], torch.Tensor):
                # Stack tensors
                try:
                    result[key] = torch.stack([sample[key] for sample in batch])
                except:
                    # If stacking fails (e.g., different shapes), use a list
                    result[key] = [sample[key] for sample in batch]
            elif isinstance(batch[0][key], np.ndarray):
                # Convert numpy arrays to tensors and stack
                try:
                    result[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch])
                except:
                    # If conversion fails, use a list
                    result[key] = [sample[key] for sample in batch]
            elif isinstance(batch[0][key], (int, float)):
                # Convert scalars to tensor
                result[key] = torch.tensor([sample[key] for sample in batch])
            else:
                # For other types, keep as list
                result[key] = [sample[key] for sample in batch]
        
        return result