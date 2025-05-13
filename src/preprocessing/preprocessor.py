import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List, Union
from omegaconf import DictConfig

from utils.utils import get_logger
from src.data.utils import convert_to_uint8, normalize_image

class PreprocessingPipeline:
    """
    Preprocessing Pipeline
    
    A configurable pipeline for image preprocessing operations.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)

        self.resize_enabled = cfg.get("resize_enabled", True)
        self.logger.info(f"Resize enabled: {self.resize_enabled}")
        self.resize_shape = tuple(cfg.get("resize_shape", [224, 224]))
        self.keep_aspect_ratio = cfg.get("keep_aspect_ratio", True)
        self.logger.info(f"Keep aspect ratio: {self.keep_aspect_ratio}")
        self.normalize = cfg.get("normalize", True)
        self.to_grayscale = cfg.get("to_grayscale", False)
        self.equalize_hist = cfg.get("equalize_hist", False)
        
        self.apply_clahe = cfg.get("apply_clahe", False)
        self.clahe_clip_limit = cfg.get("clahe_clip_limit", 2.0)
        self.clahe_tile_grid_size = tuple(cfg.get("clahe_tile_grid_size", [8, 8]))
        
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size
            )
        
        self.logger.info("PreprocessingPipeline initialized")
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process an image through the preprocessing pipeline.
        Author: Chen YANG, helped by ChatGPT
        Args:
            image: Input image (RGB format, floating point values in [0, 1])
            
        Returns:
            Processed image
        """
        original_dtype = image.dtype
        is_float = np.issubdtype(original_dtype, np.floating)
        
        img = image.copy()
        
        if is_float and np.max(img) > 1.0:
            img = img / 255.0
        
        img_uint8 = None
        if not is_float:
            img_uint8 = img.astype(np.uint8)
        else:
            img_uint8 = (img * 255).astype(np.uint8)
        
        if self.to_grayscale:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
                img_uint8 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        if self.equalize_hist:
            if len(img_uint8.shape) == 3:
                img_ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
                img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
                img_uint8 = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
            else:
                img_uint8 = cv2.equalizeHist(img_uint8)
        
        if self.apply_clahe:
            if len(img_uint8.shape) == 3:
                for i in range(img_uint8.shape[2]):
                    img_uint8[:, :, i] = self.clahe.apply(img_uint8[:, :, i])
            else:
                img_uint8 = self.clahe.apply(img_uint8)
        
        if self.resize_enabled:
            img_uint8 = self._resize_image(img_uint8)
        
        if is_float:
            img = img_uint8.astype(np.float32) / 255.0
        else:
            img = img_uint8
            
        # Apply normalization if requested
        if self.normalize and is_float:
            # Already in [0, 1] range
            pass
        
        return img
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image according to configuration.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_h, target_w = self.resize_shape
        
        if self.keep_aspect_ratio:
            scale = min(target_h / h, target_w / w)
            
            new_h, new_w = int(h * scale), int(w * scale)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            if len(image.shape) == 3:
                result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            else:
                result = np.zeros((target_h, target_w), dtype=image.dtype)
            
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            
            if len(image.shape) == 3:
                result[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
            else:
                result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
                
            return result
        else:
            return cv2.resize(image, (target_w, target_h))
    
