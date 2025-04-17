"""
Data processing module for Q-Transformer.
Provides functionality for loading, processing, and managing data.
"""

from .base import BaseDataModule, BaseDataset, DataStats
from .qtransformer_data import QTransformerDataset, QTransformerDataModule, VectorCache
from .splitters import SplitStrategy, RandomSplit, TemporalSplit, PerformanceSplit, CrossValidationSplit, create_split_strategy
from .enhanced_splitter import EnhancedSplitManager
from .utils import (
    load_npz_data, save_npz_data, create_data_visualizations,
    create_scaler, apply_scaler, save_scaler, load_scaler,
    analyze_dataset_statistics, sample_data, find_action_by_rho, compute_action_statistics
)

__all__ = [
    # Base classes
    'BaseDataModule',
    'BaseDataset',
    'DataStats',
    
    # Q-Transformer specific classes
    'QTransformerDataset',
    'QTransformerDataModule',
    'VectorCache',
    
    # Splitting strategies
    'SplitStrategy',
    'RandomSplit',
    'TemporalSplit',
    'PerformanceSplit',
    'CrossValidationSplit',
    'create_split_strategy',
    
    # Utility functions
    'load_npz_data',
    'save_npz_data',
    'create_data_visualizations',
    'create_scaler',
    'apply_scaler',
    'save_scaler',
    'load_scaler',
    'analyze_dataset_statistics',
    'sample_data',
    'find_action_by_rho',
    'compute_action_statistics'
]

# Log available classes and functions
import logging
logger = logging.getLogger(__name__)
logger.info(f"Loaded Q-Transformer data module with {len(__all__)} components")