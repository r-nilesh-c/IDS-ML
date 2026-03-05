"""
Utility functions for the Hybrid IDS.

This module provides helper functions for configuration loading,
logging setup, and reproducibility.
"""

import yaml
import logging
import logging.config
import random
import numpy as np
import os
from typing import Dict, Any


def load_config(config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(config_path: str = "config/logging_config.yaml") -> None:
    """
    Set up logging configuration from YAML file.
    
    Args:
        config_path: Path to logging configuration file
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            log_config = yaml.safe_load(f)
        logging.config.dictConfig(log_config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.warning(f"Logging config not found at {config_path}, using basic config")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Set deterministic operations for TensorFlow
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass
    
    logging.info(f"Random seeds set to {seed}")


def ensure_directories() -> None:
    """
    Create necessary directories if they don't exist.
    """
    directories = [
        "data",
        "models",
        "logs",
        "reports",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logging.info("Directory structure verified")
