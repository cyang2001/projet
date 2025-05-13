import os
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List, Union
from omegaconf import DictConfig
import glob
import matplotlib.pyplot as plt
from pathlib import Path

from utils.utils import get_logger, ensure_dir
from src.data.utils import convert_to_uint8, resize_image
from src.classification.base import BaseClassifier
from skimage.metrics import structural_similarity as ssim
class TemplateClassifier(BaseClassifier):
    """
    Template Matching Classifier
    
    Uses OpenCV template matching to classify metro line signs.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize template classifier.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        super().__init__(cfg, logger)
        self.templates = {}
        self.template_size = tuple(cfg.template.get("template_size", [64, 64]))
        self.template_dir = cfg.template.get("template_dir", "models/templates")
        self.method = eval(cfg.template.get("method", "cv2.TM_CCORR_NORMED"))
        self.threshold = cfg.template.get("threshold", 0.7)
        
        # Try to load existing templates
        self._load_templates()
        
    def _load_templates(self) -> None:
        """
        Load templates from the template directory.
        """
        if not os.path.exists(self.template_dir):
            self.logger.warning(f"Template directory does not exist: {self.template_dir}")
            return
            
        self.logger.info(f"Loading templates from {self.template_dir}")
        
        # Get all template files
        template_files = []
        for ext in ['png', 'jpg', 'jpeg']:
            template_files.extend(list(Path(self.template_dir).glob(f"*.{ext}")))
        
        if not template_files:
            self.logger.warning("No template files found")
            return
            
        # Load each template
        for template_path in template_files:
            try:
                # Extract class ID from filename (format: class_X.png)
                filename = os.path.basename(template_path)
                if not filename.startswith("class_"):
                    continue
                    
                class_id = int(filename.split("_")[1].split(".")[0])
                
                # Load and process template
                template = cv2.imread(str(template_path))
                if template is None:
                    self.logger.warning(f"Failed to load template: {template_path}")
                    continue
                    
                # Convert to RGB and normalize
                template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
                template = template.astype(np.float32) / 255.0
                    
                # Resize if needed
                if template.shape[:2] != self.template_size:
                    template = cv2.resize(template, self.template_size[::-1])  # Note: cv2.resize takes (width, height)
                
                # Store template
                self.templates[class_id] = template
                self.logger.info(f"Loaded template for class {class_id}")
                
            except Exception as e:
                self.logger.error(f"Error loading template {template_path}: {e}")
        
        self.logger.info(f"Loaded {len(self.templates)} templates")
    
    def predict(self, image: np.ndarray) -> Tuple[int, Union[float, np.floating[Any]]]:
        """
        Predict class for an image.
        
        Args:
            image: Image to classify (preprocessed ROI)
            
        Returns:
            Tuple of (class_id, confidence)
        """
        if not self.templates:
            self.logger.warning("No templates available for matching")
            return -1, 0.0
            
        # Resize input image if needed
        if image.shape[:2] != self.template_size:
            image = cv2.resize(image, self.template_size[::-1])  # Note: cv2.resize takes (width, height)
        
        # Perform template matching for each template
        best_match = -1
        best_score = -float('inf') if self.method in [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED] else float('inf')
        
        for class_id, template in self.templates.items():
            # Ensure template and image have the same dimensions
            if template.shape[:2] != image.shape[:2]:
                continue
                
            try:
                # Match templates
                if len(image.shape) == 3 and len(template.shape) == 3:
                    # For color images, match each channel separately
                    scores = []
                    for c in range(min(image.shape[2], template.shape[2])):
                        score = cv2.matchTemplate(
                            image[:, :, c].astype(np.float32), 
                            template[:, :, c].astype(np.float32), 
                            self.method
                        )[0, 0]
                        scores.append(score)
                    score = np.mean(scores)
                else:
                    # For grayscale images
                    score = cv2.matchTemplate(
                        image.astype(np.float32), 
                        template.astype(np.float32), 
                        self.method
                    )[0, 0]
                
                # Update best match
                if (self.method in [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED] and score > best_score) or \
                   (self.method not in [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED] and score < best_score):
                    best_score = score
                    best_match = class_id
                    
            except Exception as e:
                self.logger.warning(f"Error matching template for class {class_id}: {e}")
        
        # Convert score to confidence (0-1 range)
        confidence = best_score 
        
        # Apply threshold
        if confidence < self.threshold:
            return -1, confidence
            
        return best_match, confidence
    def _compute_ssim_score(self, template: np.ndarray, image: np.ndarray) -> float:
        return ssim(template, image)
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Create templates from training data.
        
        Args:
            X_train: Training images
            y_train: Training labels
        """
        self.logger.info("Creating templates from training data")
        
        if len(X_train) != len(y_train):
            self.logger.error(f"Data length mismatch: {len(X_train)} images, {len(y_train)} labels")
            return
            
        # Create template directory if it doesn't exist
        ensure_dir(self.template_dir)
        
        # Group images by class
        class_images = {}
        for i, (image, label) in enumerate(zip(X_train, y_train)):
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(image)
        
        # Create templates for each class
        for class_id, images in class_images.items():
            self.logger.info(f"Creating template for class {class_id} from {len(images)} images")
            
            # Resize all images
            resized_images = []
            for img in images:
                if img.shape[:2] != self.template_size:
                    resized = cv2.resize(img, self.template_size[::-1])
                else:
                    resized = img.copy()
                resized_images.append(resized)
            
            if not resized_images:
                continue
                
            # Average images to create template
            template = np.mean(np.array(resized_images), axis=0)
            
            # Save template
            template_path = os.path.join(self.template_dir, f"class_{class_id}.png")
            
            # Convert to 0-255 range for saving
            save_template = (template * 255).astype(np.uint8)
            
            # Convert to BGR for OpenCV
            if len(save_template.shape) == 3 and save_template.shape[2] == 3:
                save_template = cv2.cvtColor(save_template, cv2.COLOR_RGB2BGR)
                
            cv2.imwrite(template_path, save_template)
            
            # Store template in memory
            self.templates[class_id] = template
            
            self.logger.info(f"Template saved to {template_path}")
        
        self.logger.info(f"Created {len(class_images)} templates")
        
    def save(self, path: str) -> None:
        """
        Save the classifier model to disk.
        
        For template classifier, this is already done in train().
        """
        self.logger.info(f"Templates already saved to {self.template_dir}")
        
    def load(self, path: str) -> None:
        """
        Load the classifier model from disk.
        
        For template classifier, call _load_templates().
        """
        self.logger.info(f"Loading templates from {path}")
        self.template_dir = path
        self._load_templates()
        
    def visualize_templates(self) -> None:
        """
        Visualize all templates.
        """
        if not self.templates:
            self.logger.warning("No templates to visualize")
            return
            
        n_templates = len(self.templates)
        cols = min(5, n_templates)
        rows = (n_templates + cols - 1) // cols
        
        plt.figure(figsize=(cols * 3, rows * 3))
        
        for i, (class_id, template) in enumerate(sorted(self.templates.items())):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(template)
            plt.title(f"Class {class_id}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show() 