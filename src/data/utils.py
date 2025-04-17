"""
Utility functions for data processing and management.
Provides common functionality for loading, processing, and analyzing data.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
import logging
import json
import time
from pathlib import Path
import base64
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

def load_npz_data(file_path: str, allow_pickle: bool = True) -> Dict[str, np.ndarray]:
    """
    Load data from npz file.
    
    Args:
        file_path: Path to the npz file
        allow_pickle: Allow loading pickled objects
        
    Returns:
        Dictionary containing the loaded data arrays
    """
    try:
        start_time = time.time()
        data = np.load(file_path, allow_pickle=allow_pickle)
        
        # Convert to dictionary for easier access
        data_dict = {key: data[key] for key in data.files}
        
        load_time = time.time() - start_time
        logger.info(f"Loaded {file_path} in {load_time:.2f} seconds")
        logger.info(f"Found keys: {list(data_dict.keys())}")
        
        # Log data shapes
        for key, array in data_dict.items():
            shape_str = str(array.shape) if hasattr(array, 'shape') else "unknown shape"
            logger.info(f"  {key}: {shape_str}, {array.dtype}")
        
        return data_dict
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def save_npz_data(data: Dict[str, np.ndarray], file_path: str, compressed: bool = True) -> None:
    """
    Save data to npz file.
    
    Args:
        data: Dictionary containing the data arrays
        file_path: Path to save the npz file
        compressed: Whether to use compression
    """
    try:
        start_time = time.time()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save data
        if compressed:
            np.savez_compressed(file_path, **data)
        else:
            np.savez(file_path, **data)
        
        save_time = time.time() - start_time
        logger.info(f"Saved data to {file_path} in {save_time:.2f} seconds")
        
        # Log file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        logger.info(f"File size: {file_size:.2f} MB")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def create_data_visualizations(data: Dict[str, np.ndarray], output_dir: str) -> None:
    """
    Create visualizations for the dataset.
    
    Args:
        data: Data dictionary
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating data visualizations in {output_dir}")
    
    # Set figure style
    sns.set(style="whitegrid")
    
    # Visualize observation distribution
    if 'obs' in data:
        try:
            # Sample a subset of the data for visualization
            sample_size = min(10000, len(data['obs']))
            indices = np.random.choice(len(data['obs']), sample_size, replace=False)
            obs_sample = data['obs'][indices]
            
            # Compute statistics
            obs_mean = np.mean(obs_sample, axis=0)
            obs_std = np.std(obs_sample, axis=0)
            
            # Plot histogram of means
            plt.figure(figsize=(10, 6))
            plt.hist(obs_mean, bins=50)
            plt.title('Distribution of Feature Means')
            plt.xlabel('Mean Value')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, 'obs_means_histogram.png'))
            plt.close()
            
            # Plot histogram of standard deviations
            plt.figure(figsize=(10, 6))
            plt.hist(obs_std, bins=50)
            plt.title('Distribution of Feature Standard Deviations')
            plt.xlabel('Standard Deviation')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, 'obs_stds_histogram.png'))
            plt.close()
            
            logger.info("Created observation distribution visualizations")
        except Exception as e:
            logger.error(f"Error creating observation visualizations: {e}")
    
    # Visualize rho distribution
    if 'act_rho' in data:
        try:
            # Extract rho values
            rho_values = []
            sample_size = min(1000, len(data['act_rho']))
            indices = np.random.choice(len(data['act_rho']), sample_size, replace=False)
            
            for idx in indices:
                rho_data = data['act_rho'][idx]
                # Handle structured array format
                if hasattr(rho_data, 'dtype') and rho_data.dtype.names and 'rho_max' in rho_data.dtype.names:
                    rho_values.extend(rho_data['rho_max'])
                else:
                    # Handle tuple format
                    for item in rho_data:
                        if isinstance(item, tuple):
                            rho_values.append(item[1])
                        else:
                            rho_values.append(item)
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(rho_values, bins=50, kde=True)
            plt.title('Distribution of Rho Values')
            plt.xlabel('Rho Value')
            plt.ylabel('Frequency')
            plt.axvline(x=1.0, color='r', linestyle='--', label='Rho = 1.0')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'rho_distribution.png'))
            plt.close()
            
            logger.info("Created rho distribution visualization")
        except Exception as e:
            logger.error(f"Error creating rho visualization: {e}")
    
    # Visualize soft labels distribution
    if 'soft_labels' in data:
        try:
            # Extract soft label values
            soft_values = []
            sample_size = min(1000, len(data['soft_labels']))
            indices = np.random.choice(len(data['soft_labels']), sample_size, replace=False)
            
            for idx in indices:
                soft_data = data['soft_labels'][idx]
                # Handle structured array format
                if hasattr(soft_data, 'dtype') and soft_data.dtype.names and 'soft_score' in soft_data.dtype.names:
                    soft_values.extend(soft_data['soft_score'])
                else:
                    # Handle tuple format
                    for item in soft_data:
                        if isinstance(item, tuple):
                            soft_values.append(item[1])
                        else:
                            soft_values.append(item)
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(soft_values, bins=50, kde=True)
            plt.title('Distribution of Soft Label Values')
            plt.xlabel('Soft Label Value')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, 'soft_label_distribution.png'))
            plt.close()
            
            logger.info("Created soft label distribution visualization")
        except Exception as e:
            logger.error(f"Error creating soft label visualization: {e}")
    
    # Visualize timestep distribution if available
    if 'timestep' in data:
        try:
            timesteps = data['timestep']
            
            plt.figure(figsize=(12, 6))
            plt.hist(timesteps, bins=50)
            plt.title('Distribution of Timesteps')
            plt.xlabel('Timestep')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, 'timestep_distribution.png'))
            plt.close()
            
            logger.info("Created timestep distribution visualization")
        except Exception as e:
            logger.error(f"Error creating timestep visualization: {e}")

def create_scaler(
    observations: np.ndarray, 
    sample_size: Optional[int] = None, 
    clip_range: Optional[Tuple[float, float]] = None
) -> StandardScaler:
    """
    Create and fit a standard scaler for observations.
    
    Args:
        observations: Observation array (N, D)
        sample_size: Number of samples to use for fitting
        clip_range: Optional (min, max) range to clip observation values
        
    Returns:
        Fitted StandardScaler
    """
    # Sample a subset if needed
    if sample_size is not None and sample_size < len(observations):
        indices = np.random.choice(len(observations), sample_size, replace=False)
        obs_sample = observations[indices]
    else:
        obs_sample = observations
    
    # Clip values if range is provided
    if clip_range is not None:
        min_val, max_val = clip_range
        obs_sample = np.clip(obs_sample, min_val, max_val)
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(obs_sample)
    
    logger.info(f"Created scaler from {len(obs_sample)} observations")
    logger.info(f"Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    logger.info(f"Std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    
    return scaler

def apply_scaler(
    observations: np.ndarray, 
    scaler: StandardScaler, 
    clip_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Apply a standard scaler to observations.
    
    Args:
        observations: Observation array (N, D)
        scaler: Fitted StandardScaler
        clip_range: Optional (min, max) range to clip normalized values
        
    Returns:
        Normalized observations
    """
    # Transform observations
    normalized = scaler.transform(observations)
    
    # Clip values if range is provided
    if clip_range is not None:
        min_val, max_val = clip_range
        normalized = np.clip(normalized, min_val, max_val)
    
    return normalized

def save_scaler(scaler: StandardScaler, file_path: str) -> None:
    """
    Save a scaler to a file.
    
    Args:
        scaler: StandardScaler to save
        file_path: Path to save the scaler
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Save scaler
    with open(file_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"Saved scaler to {file_path}")

def load_scaler(file_path: str) -> StandardScaler:
    """
    Load a scaler from a file.
    
    Args:
        file_path: Path to the scaler file
        
    Returns:
        Loaded StandardScaler
    """
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    
    logger.info(f"Loaded scaler from {file_path}")
    return scaler

def analyze_dataset_statistics(data: Dict[str, np.ndarray], sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze dataset statistics.
    
    Args:
        data: Data dictionary
        sample_size: Number of samples to use for analysis
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Sample a subset if needed
    if sample_size is not None and 'obs' in data and sample_size < len(data['obs']):
        indices = np.random.choice(len(data['obs']), sample_size, replace=False)
    else:
        indices = None
    
    # Analyze observations
    if 'obs' in data:
        obs = data['obs']
        if indices is not None:
            obs = obs[indices]
        
        stats['obs'] = {
            'shape': obs.shape,
            'mean': float(np.mean(obs)),
            'std': float(np.std(obs)),
            'min': float(np.min(obs)),
            'max': float(np.max(obs)),
            'n_features': obs.shape[1]
        }
    
    # Analyze rho values
    if 'act_rho' in data:
        rho_data = data['act_rho']
        if indices is not None:
            rho_data = rho_data[indices]
        
        rho_values = []
        for item in rho_data:
            # Handle structured array format
            if hasattr(item, 'dtype') and item.dtype.names and 'rho_max' in item.dtype.names:
                rho_values.extend(item['rho_max'])
            else:
                # Handle tuple format
                for subitem in item:
                    if isinstance(subitem, tuple):
                        rho_values.append(subitem[1])
                    else:
                        rho_values.append(subitem)
        
        rho_values = np.array(rho_values)
        stats['rho'] = {
            'mean': float(np.mean(rho_values)),
            'std': float(np.std(rho_values)),
            'min': float(np.min(rho_values)),
            'max': float(np.max(rho_values)),
            'median': float(np.median(rho_values)),
            'count': len(rho_values)
        }
    
    # Analyze soft labels
    if 'soft_labels' in data:
        soft_data = data['soft_labels']
        if indices is not None:
            soft_data = soft_data[indices]
        
        soft_values = []
        for item in soft_data:
            # Handle structured array format
            if hasattr(item, 'dtype') and item.dtype.names and 'soft_score' in item.dtype.names:
                soft_values.extend(item['soft_score'])
            else:
                # Handle tuple format
                for subitem in item:
                    if isinstance(subitem, tuple):
                        soft_values.append(subitem[1])
                    else:
                        soft_values.append(subitem)
        
        soft_values = np.array(soft_values)
        stats['soft_labels'] = {
            'mean': float(np.mean(soft_values)),
            'std': float(np.std(soft_values)),
            'min': float(np.min(soft_values)),
            'max': float(np.max(soft_values)),
            'median': float(np.median(soft_values)),
            'count': len(soft_values)
        }
    
    # Analyze timesteps
    if 'timestep' in data:
        timesteps = data['timestep']
        if indices is not None:
            timesteps = timesteps[indices]
        
        stats['timestep'] = {
            'min': str(np.min(timesteps)),
            'max': str(np.max(timesteps)),
            'count': len(timesteps)
        }
    
    return stats

def sample_data(data: Dict[str, np.ndarray], sample_size: int) -> Dict[str, np.ndarray]:
    """
    Sample a subset of the data.
    
    Args:
        data: Data dictionary
        sample_size: Number of samples to extract
        
    Returns:
        Dictionary with sampled data
    """
    if 'obs' not in data:
        raise ValueError("Data must contain 'obs' key")
    
    # Get total number of samples
    num_samples = len(data['obs'])
    
    # Sample indices
    if sample_size >= num_samples:
        logger.warning(f"Sample size {sample_size} is >= data size {num_samples}. Using all data.")
        indices = np.arange(num_samples)
    else:
        indices = np.random.choice(num_samples, sample_size, replace=False)
    
    # Create sampled data dictionary
    sampled_data = {}
    for key, array in data.items():
        if hasattr(array, 'shape') and array.shape[0] == num_samples:
            sampled_data[key] = array[indices]
        else:
            # For arrays with different shape, just copy
            sampled_data[key] = array
    
    logger.info(f"Sampled {sample_size} examples from {num_samples} total")
    
    return sampled_data

def find_action_by_rho(data: Dict[str, np.ndarray], rho_range: Tuple[float, float], max_samples: int = 10) -> List[Tuple[int, int]]:
    """
    Find actions with rho values in the specified range.
    
    Args:
        data: Data dictionary
        rho_range: (min_rho, max_rho) range to search for
        max_samples: Maximum number of samples to return
        
    Returns:
        List of (sample_idx, action_idx) tuples
    """
    if 'act_rho' not in data:
        raise ValueError("Data must contain 'act_rho' key")
    
    min_rho, max_rho = rho_range
    matches = []
    
    # Search through data
    for sample_idx, rho_data in enumerate(data['act_rho']):
        # Handle structured array format
        if hasattr(rho_data, 'dtype') and rho_data.dtype.names and 'rho_max' in rho_data.dtype.names:
            rho_values = rho_data['rho_max']
            for action_idx, rho in enumerate(rho_values):
                if min_rho <= rho <= max_rho:
                    matches.append((sample_idx, action_idx))
                    if len(matches) >= max_samples:
                        return matches
        else:
            # Handle tuple format
            for action_idx, item in enumerate(rho_data):
                if isinstance(item, tuple):
                    rho = item[1]
                else:
                    rho = item
                
                if min_rho <= rho <= max_rho:
                    matches.append((sample_idx, action_idx))
                    if len(matches) >= max_samples:
                        return matches
    
    return matches

def compute_action_statistics(data: Dict[str, np.ndarray], max_samples: int = 1000) -> Dict[str, Any]:
    """
    Compute statistics about actions in the dataset.
    
    Args:
        data: Data dictionary
        max_samples: Maximum number of samples to analyze
        
    Returns:
        Dictionary of action statistics
    """
    if 'act_vect' not in data or 'act_rho' not in data:
        raise ValueError("Data must contain 'act_vect' and 'act_rho' keys")
    
    # Sample a subset for analysis
    num_samples = len(data['act_vect'])
    sample_size = min(max_samples, num_samples)
    indices = np.random.choice(num_samples, sample_size, replace=False)
    
    # Initialize statistics
    action_stats = {
        'action_counts': {},
        'rho_by_action': {},
        'unique_actions': 0
    }
    
    # Analyze action vectors
    unique_actions = set()
    
    for idx in indices:
        act_data = data['act_vect'][idx]
        
        for action_item in act_data:
            # Extract action key
            if isinstance(action_item, tuple):
                action_key = str(action_item[0])
            elif hasattr(action_item, 'dtype') and action_item.dtype.names:
                action_key = str(action_item['action'])
            else:
                # Skip if we can't identify the action
                continue
            
            # Count actions
            if action_key not in action_stats['action_counts']:
                action_stats['action_counts'][action_key] = 1
                action_stats['rho_by_action'][action_key] = []
            else:
                action_stats['action_counts'][action_key] += 1
            
            # Track unique actions
            unique_actions.add(action_key)
    
    # Get rho values for each action
    for idx in indices:
        act_rho_data = data['act_rho'][idx]
        
        # Handle structured array format
        if hasattr(act_rho_data, 'dtype') and act_rho_data.dtype.names:
            for i, action_key in enumerate(act_rho_data['action']):
                action_key = str(action_key)
                if action_key in action_stats['rho_by_action']:
                    action_stats['rho_by_action'][action_key].append(float(act_rho_data['rho_max'][i]))
        else:
            # Handle tuple format
            for action_item in act_rho_data:
                if isinstance(action_item, tuple):
                    action_key = str(action_item[0])
                    rho = float(action_item[1])
                    if action_key in action_stats['rho_by_action']:
                        action_stats['rho_by_action'][action_key].append(rho)
    
    # Compute statistics for each action
    for action_key in action_stats['rho_by_action']:
        rho_values = action_stats['rho_by_action'][action_key]
        if rho_values:
            action_stats['rho_by_action'][action_key] = {
                'count': len(rho_values),
                'mean': float(np.mean(rho_values)),
                'std': float(np.std(rho_values)),
                'min': float(np.min(rho_values)),
                'max': float(np.max(rho_values))
            }
    
    # Set unique action count
    action_stats['unique_actions'] = len(unique_actions)
    
    logger.info(f"Analyzed {sample_size} samples")
    logger.info(f"Found {action_stats['unique_actions']} unique actions")
    
    return action_stats