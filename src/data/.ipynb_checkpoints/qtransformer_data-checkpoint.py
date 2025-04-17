"""
Q-Transformer specific data module and dataset implementations.
Optimized for high-performance data loading and preprocessing.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
import time
import math
import base64
from pathlib import Path
import pickle
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit

from .base import BaseDataModule, BaseDataset, DataStats

# Configure logging
logger = logging.getLogger(__name__)


class VectorCache:
    """
    Thread-safe cache for storing preprocessed vectors.
    Implements LRU (Least Recently Used) eviction policy.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the vector cache.
        
        Args:
            capacity: Maximum number of entries in the cache
        """
        self.capacity = capacity
        self.cache = {}  # Key -> (value, timestamp)
        self.lock = threading.RLock()
        self.access_count = 0
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached tensor or None if not found
        """
        with self.lock:
            if key in self.cache:
                value, _ = self.cache[key]
                # Update timestamp
                self.access_count += 1
                self.cache[key] = (value, self.access_count)
                return value
        return None
    
    def put(self, key: str, value: torch.Tensor) -> None:
        """
        Put a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.capacity and key not in self.cache:
                # Find least recently used entry
                oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
                self.cache.pop(oldest_key)
            
            # Update access count
            self.access_count += 1
            self.cache[key] = (value, self.access_count)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_count = 0
    
    def info(self) -> Dict[str, Any]:
        """
        Get information about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            return {
                'size': len(self.cache),
                'capacity': self.capacity,
                'usage': len(self.cache) / self.capacity if self.capacity > 0 else 0,
                'access_count': self.access_count
            }


class QTransformerDataset(BaseDataset):
    """
    Dataset for Q-Transformer model.
    Optimized for efficient loading and preprocessing of Q-Transformer data.
    """
    
    def __init__(
        self, 
        data: Dict[str, np.ndarray], 
        indices: Optional[np.ndarray] = None,
        config: Dict[str, Any] = None,
        is_training: bool = True,
        prefetch: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Dictionary containing the data arrays
            indices: Indices to use from the data (if None, use all)
            config: Configuration dictionary
            is_training: Whether this dataset is for training
            prefetch: Whether to prefetch and cache data
        """
        super().__init__(config or {}, is_training)
        self.data = data
        
        # Set indices to use for this dataset
        if indices is None:
            self.indices = np.arange(len(data['obs']))
        else:
            self.indices = indices
        
        # Configure vector caching
        cache_capacity = config.get('cache_capacity', 100000)
        self.action_cache = VectorCache(cache_capacity)
        self.soft_label_cache = VectorCache(cache_capacity)
        
        # Determine dimensions and structure
        self._analyze_data_structure()
        
        # Prefetch data if enabled
        if prefetch:
            self._prefetch_data()
    
    def _analyze_data_structure(self):
        """Analyze the structure of the data to determine processing approach."""
        # Get sample shape for obs
        if len(self.data['obs']) > 0:
            self.obs_dim = self.data['obs'][0].shape[0]
        else:
            self.obs_dim = 0
            logger.warning("No observation data found")
        
        # Determine number of actions
        if 'act_rho' in self.data and len(self.data['act_rho']) > 0:
            self.n_actions = len(self.data['act_rho'][0])
        else:
            self.n_actions = self.config.get('n_actions', 50)
            logger.warning(f"No action data found, using config value: {self.n_actions}")
        
        # Determine action vector dimension
        if 'act_vect' in self.data and len(self.data['act_vect']) > 0:
            sample_action = self.data['act_vect'][0]
            
            # Check action vector structure
            if sample_action.ndim == 2 and sample_action.shape[1] == 2:
                # Format: [n_actions, 2] where each row is [key, vector]
                action_vector = sample_action[0, 1]
                if hasattr(action_vector, 'shape'):
                    self.action_dim = action_vector.shape[0]
                else:
                    # Try to determine from the vector itself
                    self.action_dim = len(action_vector)
            elif isinstance(sample_action[0], tuple) or hasattr(sample_action[0], 'dtype'):
                # Structured array or tuple format
                if isinstance(sample_action[0], tuple):
                    _, action_vector = sample_action[0]
                else:
                    action_vector = sample_action[0]['act_vect']
                
                self.action_dim = action_vector.shape[0] if hasattr(action_vector, 'shape') else len(action_vector)
            else:
                # Default
                self.action_dim = self.config.get('action_dim', 1152)
                logger.warning(f"Could not determine action dimension, using config value: {self.action_dim}")
        else:
            self.action_dim = self.config.get('action_dim', 1152)
            logger.warning(f"No action vector data found, using config value: {self.action_dim}")
        
        logger.info(f"Dataset dimensions - Observations: {self.obs_dim}, Actions: {self.n_actions}, Action vector: {self.action_dim}")
    
    def _prefetch_data(self):
        """Prefetch and cache data for faster access."""
        logger.info("Prefetching data for faster access...")
        start_time = time.time()
        
        # Sample indices for prefetching (to avoid memory overload)
        max_prefetch = min(1000, len(self.indices))
        
        # Sửa: Sử dụng các chỉ số trực tiếp từ self.indices thay vì tạo các chỉ số mới
        sample_indices = np.random.choice(len(self.indices), max_prefetch, replace=False)
        prefetch_indices = [self.indices[i] for i in sample_indices]
        
        # Process each sample
        for idx in prefetch_indices:
            # Sửa: Gọi _process_* trực tiếp thay vì thông qua __getitem__
            self._process_observation(idx)
            self._process_action_vectors(idx)
            self._process_rho_values(idx)
            self._process_soft_labels(idx)
        
        logger.info(f"Data prefetching completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Action cache stats: {self.action_cache.info()}")
        logger.info(f"Soft label cache stats: {self.soft_label_cache.info()}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        # Get the actual index in the data
        actual_idx = self.indices[idx]
        
        # Process observation
        observation = self._process_observation(actual_idx)
        
        # Process action vectors
        action_vectors = self._process_action_vectors(actual_idx)
        
        # Process rho values
        rho_values = self._process_rho_values(actual_idx)
        
        # Process soft labels
        soft_labels = self._process_soft_labels(actual_idx)
        
        # Get timestep if available
        timestep = None
        if 'timestep' in self.data:
            timestep = self.data['timestep'][actual_idx]
        
        # Create sample dictionary
        sample = {
            'observation': observation,
            'action_vectors': action_vectors,
            'rho_values': rho_values,
            'soft_labels': soft_labels,
            'timestep': timestep,
            'idx': actual_idx
        }
        
        # Add best action if available
        if 'best_action' in self.data:
            sample['best_action'] = self.data['best_action'][actual_idx]
        
        return sample
    
    def _process_observation(self, idx: int) -> torch.Tensor:
        """
        Process observation vector.
        
        Args:
            idx: Index of the observation
            
        Returns:
            Processed observation tensor
        """
        observation = self.data['obs'][idx]
        
        # Convert to tensor
        observation = torch.FloatTensor(observation)
        
        # Apply normalization if enabled
        if self.vector_scaler:
            observation = self.normalize_observation(observation)
        
        return observation
    
    def _process_action_vectors(self, idx: int) -> torch.Tensor:
        """
        Process action vectors for a sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tensor containing all action vectors
        """
        action_vectors = []
        action_data = self.data['act_vect'][idx]
        
        # Process based on structure
        if action_data.ndim == 2 and action_data.shape[1] == 2:
            # Format: [n_actions, 2] array with [key, vector] rows
            for i in range(len(action_data)):
                key = str(action_data[i, 0])
                
                # Check cache
                cached_vector = self.action_cache.get(key)
                if cached_vector is not None:
                    action_vectors.append(cached_vector)
                    continue
                
                # Process vector
                vector = action_data[i, 1]
                if hasattr(vector, 'shape'):
                    vector_tensor = torch.FloatTensor(vector)
                else:
                    # Handle other formats
                    vector_tensor = torch.FloatTensor(list(vector))
                
                # Cache for future use
                self.action_cache.put(key, vector_tensor)
                action_vectors.append(vector_tensor)
        else:
            # Assume structured array or tuple format
            for action_item in action_data:
                if isinstance(action_item, tuple):
                    key = str(action_item[0])
                    vector = action_item[1]
                elif hasattr(action_item, 'dtype') and action_item.dtype.names:
                    # Structured array
                    key = str(action_item['action'])
                    vector = action_item['act_vect']
                else:
                    # Direct vector
                    action_vectors.append(torch.FloatTensor(action_item))
                    continue
                
                # Check cache
                cached_vector = self.action_cache.get(key)
                if cached_vector is not None:
                    action_vectors.append(cached_vector)
                    continue
                
                # Process vector
                if hasattr(vector, 'shape'):
                    vector_tensor = torch.FloatTensor(vector)
                else:
                    vector_tensor = torch.FloatTensor(list(vector))
                
                # Cache for future use
                self.action_cache.put(key, vector_tensor)
                action_vectors.append(vector_tensor)
        
        # Stack tensors
        if action_vectors:
            return torch.stack(action_vectors)
        else:
            # Return empty tensor with correct shape
            return torch.zeros((self.n_actions, self.action_dim))
    
    def _process_rho_values(self, idx: int) -> torch.Tensor:
        """
        Process rho_max values for a sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tensor of rho_max values
        """
        rho_values = []
        rho_data = self.data['act_rho'][idx]
        
        # Handle structured array format (most common)
        if hasattr(rho_data, 'dtype') and rho_data.dtype.names and 'rho_max' in rho_data.dtype.names:
            rho_values = rho_data['rho_max'].astype(np.float32)
        else:
            # Handle tuple format
            for action_item in rho_data:
                if isinstance(action_item, tuple):
                    rho_values.append(float(action_item[1]))
                else:
                    # Direct value
                    rho_values.append(float(action_item))
        
        return torch.FloatTensor(rho_values)
    
    def _process_soft_labels(self, idx: int) -> torch.Tensor:
        """
        Process soft labels for a sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tensor of soft labels
        """
        soft_values = []
        soft_data = self.data['soft_labels'][idx]
        
        # Process based on structure
        if hasattr(soft_data, 'dtype') and soft_data.dtype.names and 'soft_score' in soft_data.dtype.names:
            # Structured array with 'soft_score' field
            soft_values = soft_data['soft_score'].astype(np.float32)
        else:
            # Handle tuple or other formats
            for label_item in soft_data:
                if isinstance(label_item, tuple):
                    key = str(label_item[0])
                    
                    # Check cache
                    cached_value = self.soft_label_cache.get(key)
                    if cached_value is not None:
                        soft_values.append(cached_value.item())
                        continue
                    
                    # Get soft score
                    soft_score = float(label_item[1])
                    
                    # Cache for future use
                    self.soft_label_cache.put(key, torch.tensor(soft_score))
                    soft_values.append(soft_score)
                else:
                    # Direct value
                    soft_values.append(float(label_item))
        
        # Convert to tensor
        soft_tensor = torch.FloatTensor(soft_values)
        
        # Ensure probabilities sum to 1
        if soft_tensor.sum() > 0:
            soft_tensor = soft_tensor / soft_tensor.sum()
        else:
            # If all zero, use uniform distribution
            soft_tensor = torch.ones_like(soft_tensor) / len(soft_tensor)
        
        return soft_tensor
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with detailed sample information
        """
        # Get the actual index in the data
        actual_idx = self.indices[idx]
        
        # Process sample
        sample = self[idx]
        
        # Compute additional information
        info = {
            'idx': actual_idx,
            'observation_shape': sample['observation'].shape,
            'observation_range': (float(sample['observation'].min()), float(sample['observation'].max())),
            'observation_mean': float(sample['observation'].mean()),
            'observation_std': float(sample['observation'].std()),
            'action_vectors_shape': sample['action_vectors'].shape,
            'rho_values_range': (float(sample['rho_values'].min()), float(sample['rho_values'].max())),
            'rho_values_mean': float(sample['rho_values'].mean()),
            'soft_labels_entropy': float(-torch.sum(sample['soft_labels'] * torch.log(sample['soft_labels'] + 1e-10))),
            'timestep': sample['timestep']
        }
        
        return info


class QTransformerDataModule(BaseDataModule):
    """
    Data module for Q-Transformer model.
    Handles loading, preprocessing, and splitting of datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data module.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize datasets to None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Get split configuration
        self.val_size = config.get('val_size', 0.2)
        self.split_strategy = config.get('split_strategy', 'random')
        self.random_seed = config.get('random_seed', 42)
    
    def setup(self):
        """Set up the data module, loading and processing datasets."""
        # Load training data
        train_data = self._load_training_files()
        
        # Split into train and validation
        train_indices, val_indices = self._split_train_val(train_data)
        
        # Create datasets
        logger.info("Creating training dataset...")
        self.train_dataset = QTransformerDataset(
            data=train_data,
            indices=train_indices,
            config=self.config,
            is_training=True
        )
        
        logger.info("Creating validation dataset...")
        self.val_dataset = QTransformerDataset(
            data=train_data,
            indices=val_indices,
            config=self.config,
            is_training=False
        )
        
        # Load test data if available
        if self.config.get('testing_files'):
            logger.info("Loading test data...")
            test_data = self._load_testing_files()
            
            # Log info about external test dataset
            logger.info(f"External test dataset contains {len(test_data['obs'])} samples")
            
            if 'timestep' in test_data:
                test_times = test_data['timestep']
                logger.info(f"Test time range: {np.min(test_times)} → {np.max(test_times)}")
            
            # Tiếp tục code hiện có
            logger.info("Creating test dataset...")
            self.test_dataset = QTransformerDataset(
                data=test_data,
                indices=None,  # Use all data
                config=self.config,
                is_training=False
            )
        
        # Compute and cache statistics
        if self.config.get('vector_scaler', False):
            logger.info("Computing dataset statistics...")
            self.stats = self.compute_stats([self.train_dataset, self.val_dataset])
    
    def _load_training_files(self) -> Dict[str, np.ndarray]:
        """
        Load training data from files.
        
        Returns:
            Dictionary containing training data
        """
        training_files = self.config.get('training_files', [])
        if not training_files:
            raise ValueError("No training files specified in config")
        
        logger.info(f"Loading training data from {len(training_files)} files...")
        
        # Load each file
        data_list = []
        for file_name in training_files:
            file_path = self.dataset_dir / file_name
            logger.info(f"Loading file: {file_path}")
            
            # Load data
            data = dict(np.load(file_path, allow_pickle=True))
            data_list.append(data)
        
        # Combine data if multiple files
        if len(data_list) > 1:
            logger.info("Combining data from multiple files...")
            return self._combine_data(data_list)
        else:
            return data_list[0]
    
    def _load_testing_files(self) -> Dict[str, np.ndarray]:
        """
        Load testing data from files.
        
        Returns:
            Dictionary containing testing data
        """
        testing_files = self.config.get('testing_files', [])
        if not testing_files:
            logger.warning("No testing files specified in config")
            return {}
        
        logger.info(f"Loading testing data from {len(testing_files)} files...")
        
        # Load each file
        data_list = []
        for file_name in testing_files:
            file_path = self.dataset_dir / file_name
            logger.info(f"Loading file: {file_path}")
            
            # Load data
            data = dict(np.load(file_path, allow_pickle=True))
            data_list.append(data)
        
        # Combine data if multiple files
        if len(data_list) > 1:
            logger.info("Combining data from multiple files...")
            return self._combine_data(data_list)
        else:
            return data_list[0]
    
    def _combine_data(self, data_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Combine data from multiple sources.
        
        Args:
            data_list: List of data dictionaries to combine
            
        Returns:
            Combined data dictionary
        """
        combined_data = {}
        
        # Get all keys
        all_keys = set()
        for data in data_list:
            all_keys.update(data.keys())
        
        # Combine each key
        for key in all_keys:
            # Get arrays for this key
            arrays = [data[key] for data in data_list if key in data]
            
            if not arrays:
                continue
            
            # Check if all arrays have the same shape[1:]
            first_shape = arrays[0].shape[1:] if arrays[0].shape else None
            if not all(arr.shape[1:] == first_shape for arr in arrays):
                logger.warning(f"Arrays for key {key} have different shapes. Skipping...")
                continue
            
            # Concatenate along axis 0
            try:
                combined_data[key] = np.concatenate(arrays, axis=0)
                logger.info(f"Combined data for key {key}: shape={combined_data[key].shape}")
            except Exception as e:
                logger.error(f"Error combining data for key {key}: {e}")
        
        return combined_data
    
    def _split_train_val(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets using the enhanced splitter.
        """
        if 'obs' not in data:
            raise ValueError("Data must contain 'obs' key")
        
        # Create enhanced split manager with test_size=0
        from .enhanced_splitter import EnhancedSplitManager
        
        # Override test_size to 0 since we're using a separate test file
        config_copy = self.config.copy()
        config_copy['test_size'] = 0
        
        splitter = EnhancedSplitManager(
            config=config_copy,
            visualization_dir=os.path.join(self.cache_dir, 'visualizations')
        )
        
        # Perform split - this will only split train/val
        train_indices, val_indices, _ = splitter.split_data(data)
        
        # Save the report
        splitter.save_report()
        
        return train_indices, val_indices
    
    def get_train_dataset(self) -> Dataset:
        """
        Get the training dataset.
        
        Returns:
            Training dataset
        """
        if self.train_dataset is None:
            raise RuntimeError("Data module not set up. Call setup() first.")
        return self.train_dataset
    
    def get_val_dataset(self) -> Dataset:
        """
        Get the validation dataset.
        
        Returns:
            Validation dataset
        """
        if self.val_dataset is None:
            raise RuntimeError("Data module not set up. Call setup() first.")
        return self.val_dataset
    
    def get_test_dataset(self) -> Dataset:
        """
        Get the test dataset.
        
        Returns:
            Test dataset
        """
        if self.test_dataset is None:
            logger.warning("Test dataset not available.")
            return None
        return self.test_dataset
    
    def get_train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """
        Get the training data loader.
        
        Args:
            batch_size: Batch size (overrides config if provided)
            
        Returns:
            Training data loader
        """
        dataset = self.get_train_dataset()
        batch_size = batch_size or self.config.get('training_params', {}).get('batch_size', 32)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )
    
    def get_val_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """
        Get the validation data loader.
        
        Args:
            batch_size: Batch size (overrides config if provided)
            
        Returns:
            Validation data loader
        """
        dataset = self.get_val_dataset()
        batch_size = batch_size or self.config.get('training_params', {}).get('batch_size', 32)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )
    
    def get_test_dataloader(self, batch_size: Optional[int] = None) -> Optional[DataLoader]:
        """
        Get the test data loader.
        
        Args:
            batch_size: Batch size (overrides config if provided)
            
        Returns:
            Test data loader or None if not available
        """
        dataset = self.get_test_dataset()
        if dataset is None:
            return None
        
        batch_size = batch_size or self.config.get('training_params', {}).get('batch_size', 32)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )