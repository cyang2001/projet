import os
import json
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from omegaconf import DictConfig

from utils.utils import get_logger, ensure_dir
from src.data.dataset import MetroDataset

class ROIDatasetGenerator:
    """
    Tool for extracting ROIs from original dataset and creating a dedicated dataset for classification.
    
    Extracts ROIs based on annotations, applies preprocessing, and saves the results
    along with metadata in JSON format.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the ROI dataset generator.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.cfg = cfg
        print(self.cfg)
        
        # Get configuration parameters
        self.output_dir = self.cfg.get("roi_dataset_dir", os.path.join(self.cfg.get("output_dir", "results"), "roi_dataset"))
        self.add_background_samples = self.cfg.get("add_background_samples", True)
        self.background_class_id = self.cfg.get("background_class_id", -1)
        self.num_bg_samples_per_image = self.cfg.get("num_bg_samples_per_image", 2)
        self.random_seed = self.cfg.get("random_seed", 42)
        
        # Setup paths
        self.roi_dir = os.path.join(self.output_dir, "rois")
        self.metadata_path = os.path.join(self.output_dir, "metadata.json")
        # Preprocessor is passed separately as it might come from another component
        self.preprocessor = None
        
        # Create output directories
        ensure_dir(self.output_dir)
        ensure_dir(self.roi_dir)
        
        self.logger.info(f"ROIDatasetGenerator initialized with output_dir={self.output_dir}")
    
    def set_preprocessor(self, preprocessor):
        """
        Set the preprocessor for ROI images.
        
        Args:
            preprocessor: Preprocessor instance to apply to ROIs
        """
        self.preprocessor = preprocessor
        return self
        
    def generate_from_dataset(self, dataset: MetroDataset) -> Dict[str, Any]:
        """
        Extract ROIs from dataset and generate classification dataset.
        
        Args:
            dataset: MetroDataset instance
            
        Returns:
            Metadata dictionary with dataset information
        """
        self.logger.info(f"Generating ROI dataset from {len(dataset)} images")
        
        np.random.seed(self.random_seed)
        metadata = {
            "samples": [],
            "class_distribution": {},
            "total_samples": 0,
            "dataset_type": dataset.mode,
            "creation_params": {
                "add_background_samples": self.add_background_samples,
                "background_class_id": self.background_class_id,
                "random_seed": self.random_seed
            }
        }
        
        for idx in range(len(dataset)):
            image, annotations = dataset.get_image_with_annotations(idx)
            image_id = dataset.df.iloc[idx]['image_id']
            
            # Process each annotation (real ROI) in the image
            for ann_idx, ann in enumerate(annotations):
                x1, y1, x2, y2, class_id = ann
                
                # Extract ROI
                roi = image[y1:y2, x1:x2]
                if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                    self.logger.warning(f"Invalid ROI in image {image_id}: {ann}")
                    continue
                
                # Apply preprocessing if available
                if self.preprocessor:
                    roi = self.preprocessor.process(roi)
                # Generate unique filename for this ROI
                roi_filename = f"roi_{image_id}_{ann_idx}.png"
                roi_path = os.path.join(self.roi_dir, roi_filename)
                if roi.dtype == np.float32 or roi.dtype == np.float64:
                    roi_to_save = (roi * 255).clip(0, 255).astype(np.uint8)
                elif roi.dtype != np.uint8:
                    roi_norm = (roi - roi.min()) / (roi.max() - roi.min() + 1e-5)
                    roi_to_save = (roi_norm * 255).astype(np.uint8)
                else:
                    roi_to_save = roi
                cv2.imwrite(roi_path, roi_to_save)
                # Add to metadata
                sample_info = {
                    "id": f"{str(image_id)}_{int(ann_idx)}",
                    "image_id": str(image_id),
                    "roi_file": str(roi_filename),
                    "class_id": int(class_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "is_background": False
                }
                
                metadata["samples"].append(sample_info)
                
                # Update class distribution
                class_str = str(int(class_id))
                metadata["class_distribution"][class_str] = metadata["class_distribution"].get(class_str, 0) + 1
            
            # Generate background (negative) samples if requested
            if self.add_background_samples:
                num_samples = min(self.num_bg_samples_per_image, len(annotations)) if len(annotations) > 0 else 1
                bg_samples = self._generate_background_samples(
                    image, annotations, image_id, num_samples=num_samples
                )
                metadata["samples"].extend(bg_samples)
                
                # Update background class in distribution
                bg_class = str(self.background_class_id)
                metadata["class_distribution"][bg_class] = metadata["class_distribution"].get(bg_class, 0) + len(bg_samples)
        
        # Update total samples count
        metadata["total_samples"] = len(metadata["samples"])
        
        # Save metadata to JSON file
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Generated ROI dataset with {metadata['total_samples']} samples")
        self.logger.info(f"Class distribution: {metadata['class_distribution']}")
        
        return metadata
    
    def _generate_background_samples(self, 
                                    image: np.ndarray, 
                                    annotations: List[Tuple], 
                                    image_id: Any,
                                    num_samples: int = 2) -> List[Dict]:
        """
        Generate background (negative) samples from image regions not containing ROIs.
        
        Args:
            image: Source image
            annotations: List of annotations (x1, y1, x2, y2, class_id)
            image_id: ID of the source image
            num_samples: Number of background samples to generate
            
        Returns:
            List of background sample metadata
        """
        height, width = image.shape[:2]
        bg_samples = []
        
        # Create a mask of regions covered by annotations
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in annotations:
            x1, y1, x2, y2, _ = ann
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            mask[y1:y2, x1:x2] = 1
        
        # Get size range based on actual ROIs
        if annotations:
            sizes = [(ann[2] - ann[0], ann[3] - ann[1]) for ann in annotations]
            avg_width = sum(s[0] for s in sizes) / len(sizes)
            avg_height = sum(s[1] for s in sizes) / len(sizes)
        else:
            avg_width, avg_height = 50, 50  # Fallback default
        
        attempts = 0
        max_attempts = num_samples * 10
        bg_idx = 0
        
        while len(bg_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Random size with some variation around average ROI size
            w = int(avg_width * np.random.uniform(0.8, 1.2))
            h = int(avg_height * np.random.uniform(0.8, 1.2))
            
            # Random position
            x1 = np.random.randint(0, max(1, width - w))
            y1 = np.random.randint(0, max(1, height - h))
            x2 = min(x1 + w, width)
            y2 = min(y1 + h, height)
            
            # Check if this region overlaps with any annotation
            roi_mask = mask[y1:y2, x1:x2]
            if roi_mask.sum() > 0:
                continue  # Region overlaps with an annotation, try again
            
            # Extract background ROI
            bg_roi = image[y1:y2, x1:x2]
            if bg_roi.size == 0 or bg_roi.shape[0] == 0 or bg_roi.shape[1] == 0:
                continue
            
            # Apply preprocessing if available
            if self.preprocessor:
                bg_roi = self.preprocessor.process(bg_roi)
            
            # Generate filename
            roi_filename = f"bg_{image_id}_{bg_idx}.png"
            roi_path = os.path.join(self.roi_dir, roi_filename)
            if bg_roi.dtype == np.float32 or bg_roi.dtype == np.float64:
                bg_roi_to_save = (bg_roi * 255).clip(0, 255).astype(np.uint8)
            elif bg_roi.dtype != np.uint8:
                bg_roi_norm = (bg_roi - bg_roi.min()) / (bg_roi.max() - bg_roi.min() + 1e-5)
                bg_roi_to_save = (bg_roi_norm * 255).astype(np.uint8)
            else:
                bg_roi_to_save = bg_roi
            # Save ROI image
            cv2.imwrite(roi_path, bg_roi_to_save)
            
            # Add to metadata
            sample_info = {
                "id": f"bg_{image_id}_{int(bg_idx)}",
                "image_id": str(image_id),
                "roi_file": str(roi_filename),
                "class_id": int(self.background_class_id),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "is_background": True
            }
            
            bg_samples.append(sample_info)
            bg_idx += 1
        
        return bg_samples


class ROIDatasetLoader:
    """
    Loader for pre-extracted ROI dataset.
    
    Loads ROI images and metadata from the generated dataset.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the ROI dataset loader.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.cfg = cfg
        
        # Get configuration parameters
        self.dataset_dir = self.cfg.get("roi_dataset_dir", os.path.join(self.cfg.get("output_dir", "results"), "roi_dataset"))
        
        # Setup paths
        self.roi_dir = os.path.join(self.dataset_dir, "rois")
        self.metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        
        # Preprocessor is passed separately as it might come from another component
        self.preprocessor = None
        
        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.samples = self.metadata["samples"]
        self.class_distribution = self.metadata.get("class_distribution", {})
        
        self.logger.info(f"Loaded ROI dataset with {len(self.samples)} samples")
        self.logger.info(f"Class distribution: {self.class_distribution}")
        
    def set_preprocessor(self, preprocessor):
        """
        Set the preprocessor for ROI images.
        
        Args:
            preprocessor: Preprocessor instance to apply to ROIs
        """
        self.preprocessor = preprocessor
        return self
        
    def get_all_data(self):
        """
        Get all ROI images and labels.
        
        Returns:
            Tuple of (images array, labels array)
        """
        images = []
        labels = []
        
        for sample in self.samples:
            roi_path = os.path.join(self.roi_dir, sample["roi_file"])
            if not os.path.exists(roi_path):
                self.logger.warning(f"ROI file not found: {roi_path}")
                continue
                
            try:
                roi = cv2.imread(roi_path)
                if roi is None:
                    self.logger.warning(f"Failed to read ROI: {roi_path}")
                    continue
                    
                if self.preprocessor and not sample.get("preprocessed", False):
                    roi = self.preprocessor.process(roi)
                images.append(roi)
                labels.append(sample["class_id"])
            except Exception as e:
                self.logger.error(f"Error loading ROI {roi_path}: {e}")
        
        if not images:
            self.logger.warning("No valid images found in ROI dataset")
            return np.array([]), np.array([])
            
        return np.array(images), np.array(labels)
        
    def split_train_val(self, val_ratio=0.2, random_seed=None):
        """
        Split dataset into training and validation sets.
        
        Args:
            val_ratio: Ratio of validation samples
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        if random_seed is None:
            random_seed = self.cfg.get("random_seed", 42)
            
        np.random.seed(random_seed)
        X, y = self.get_all_data()
        
        if len(X) == 0:
            self.logger.warning("Cannot split empty dataset")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        val_size = int(len(X) * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        return X_train, y_train, X_val, y_val
        
    def get_class_balance_weights(self):
        """
        Calculate class weights to handle imbalanced data.
        
        Returns:
            Dictionary of class weights {class_id: weight}
        """
        if not self.class_distribution:
            self.logger.warning("No class distribution found in metadata")
            return {}
            
        total_samples = sum(int(count) for count in self.class_distribution.values())
        num_classes = len(self.class_distribution)
        
        weights = {}
        for class_id, count in self.class_distribution.items():
            weights[int(class_id)] = total_samples / (num_classes * int(count))
            
        return weights 