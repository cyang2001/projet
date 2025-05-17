import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import logging
from omegaconf import DictConfig

from utils.utils import get_logger

class CNNPreprocessor:
    """
    Preprocessor specifically for CNN classification of ROIs.
    
    Handles image resizing, normalization, and other preprocessing steps
    required for CNN input.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the CNN preprocessor.
        
        Args:
            cfg: Configuration object
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.cfg = cfg
        # Get preprocessing parameters from config
        self.target_size = self.cfg.get("target_size", (64, 64))
        self.normalize = self.cfg.get("normalize", True)
        self.channel_order = self.cfg.get("channel_order", "rgb")  # 'rgb' or 'bgr'
        
        self.logger.info(f"CNN Preprocessor initialized with target_size={self.target_size}")
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for CNN input.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        if image.size == 0:
            raise ValueError("Empty image provided to preprocessor")
        
        # Resize image to target size
        resized = self._resize_image(image)
        
        # Convert color space if needed
        if self.channel_order.lower() == 'bgr' and len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        # Normalize pixel values
        if self.normalize:
            resized = resized.astype(np.float32) / 255.0

        return resized
        
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        target_width, target_height = self.target_size
        
        # Ensure the image is of appropriate dimensions
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # Resize the image
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return resized 