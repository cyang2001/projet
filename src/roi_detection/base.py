"""
Base classes and factory functions for ROI detection.
"""

import json
import logging
from abc import ABC, abstractmethod
import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from omegaconf import DictConfig

from utils.utils import get_logger

class BaseDetector(ABC):
    """
    Base class for ROI detectors.
    
    All detectors must implement the detect method.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the detector.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect ROIs in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected ROIs as (x1, y1, x2, y2, confidence)
        """
        pass
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        Update detector parameters.
        
        Args:
            params: Parameter dictionary
        """
        self.logger.warning("update_params not implemented in this detector")

    def save_params(self, params_dir: str, params: Dict[str, Any]) -> bool:
        """
        Save detector parameters.
        
        Args:
            params_dir: Directory to save parameters
        """
        try:
            with open(os.path.join(params_dir, "params.json"), "w") as f:
                json.dump(params, f)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save parameters: {e}")
            return False

def get_detector(cfg: DictConfig, logger: Optional[logging.Logger] = None) -> BaseDetector:
    """
    Factory function to get ROI detector based on configuration.
    
        cfg: Configuration dictionary
        logger: Optional logger
        
    Returns:
        ROI detector instance
    """
    detector_name = cfg.get("name", "multi_color_detector")
    
    if logger:
        logger.info(f"Creating detector of type: {detector_name}")
    
    if detector_name == "multi_color_detector":
        from .multi_color_detector import MultiColorDetector
        return MultiColorDetector(cfg, logger)
    else:
        raise ValueError(f"Unsupported detector type: {detector_name}") 