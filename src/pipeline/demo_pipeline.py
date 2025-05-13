import logging
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import DictConfig
from typing import List, Optional, Tuple, Dict, Any, Union

from src.data.dataset import MetroDataset
from src.roi_detection.base import get_detector
from src.classification.base import get_classifier
from src.preprocessing.preprocessor import PreprocessingPipeline
from utils.utils import get_logger, ensure_dir

class MetroDemoPipeline:
    """
    Paris Metro Line Recognition Demo Pipeline
    
    Processes single images or batches of images and visualizes the results.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the demo pipeline.
        
        Args:
            cfg: Configuration object
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.cfg = cfg
        self._validate_config()
        self._init_components()
    
    def _validate_config(self):
        """
        Validate the configuration parameters.
        """
        required_configs = {
            'detector': "ROI detector configuration",
            'classification': "Classification configuration",
            'preprocessing': "Preprocessing configuration"
        }
        
        for key, description in required_configs.items():
            if not hasattr(self.cfg, key):
                raise ValueError(f"Configuration missing: {key} ({description})")
                
        if not hasattr(self.cfg.mode, 'demo'):
            raise ValueError("Configuration missing: mode.demo")
            
        if not hasattr(self.cfg.mode.demo, 'input_path'):
            raise ValueError("Configuration missing: mode.demo.input_path")
            
        self.logger.info("Configuration validation passed")
    
    def _init_components(self):
        """
        Initialize pipeline components.
        """
        try:
            self.logger.info("Initializing demo components...")
            
            # Initialize preprocessor
            self.preprocessor = PreprocessingPipeline(
                cfg=self.cfg.preprocessing
            )
            
            # Initialize detector
            self.detector = get_detector(
                cfg=self.cfg.detector,
                logger=self.logger
            )
            
            # Initialize classifier
            self.classifier = get_classifier(
                cfg=self.cfg.classification,
                logger=self.logger
            )
            
            self.logger.info("Demo components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run(self):
        """
        Run the demo pipeline.
        
        Process single image or all images in a directory based on configuration.
        """
        self.logger.info("=== Starting Demo Pipeline ===")
        
        input_path = self.cfg.mode.demo.input_path
        batch_mode = self.cfg.mode.demo.get("batch_mode", False)
        
        if batch_mode:
            self._process_batch(input_path)
        else:
            self._process_single(input_path)
            
        self.logger.info("=== Demo Completed ===")
    
    def _process_single(self, image_path: str):
        """
        Process a single image.
        
        Args:
            image_path: Path to the image file
        """
        import cv2
        from PIL import Image
        
        self.logger.info(f"Processing single image: {image_path}")
        
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            return
        
        try:
            # Load image
            image = np.array(Image.open(image_path).convert('RGB'))
            image_float = image.astype(np.float32) / 255.0
            
            # Preprocess
            preprocessed = self.preprocessor.process(image_float)
            
            # Detect ROIs
            rois = self.detector.detect(preprocessed)
            
            self.logger.info(f"Detected {len(rois)} potential metro signs")
            
            # Classify ROIs
            results = []
            for roi in rois:
                x1, y1, x2, y2, _ = roi
                
                # Extract ROI region
                roi_img = preprocessed[y1:y2, x1:x2]
                
                # Classify
                class_id, confidence = self.classifier.predict(roi_img)
                
                if class_id != -1:  # Valid classification
                    results.append({
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id,
                        'confidence': confidence
                    })
                    self.logger.info(f"Detected metro line {class_id} with confidence {confidence:.4f}")
            
            # Visualize results
            self._visualize_results(image, results, os.path.basename(image_path))
            
            # Save results if needed
            if self.cfg.mode.demo.get("save_results", False):
                output_dir = self.cfg.get("output_dir", "results")
                self._save_visualization(image, results, os.path.join(output_dir, f"demo_{os.path.basename(image_path)}"))
        
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
    
    def _process_batch(self, directory: str):
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
        """
        self.logger.info(f"Processing images in directory: {directory}")
        
        if not os.path.isdir(directory):
            self.logger.error(f"Directory not found: {directory}")
            return
        
        # Get all image files
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
            image_files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
        
        self.logger.info(f"Found {len(image_files)} images")
        
        if not image_files:
            self.logger.warning(f"No images found in {directory}")
            return
        
        # Process each image
        all_results = []
        for image_path in image_files:
            try:
                self.logger.info(f"Processing {os.path.basename(image_path)}")
                
                # Use simplified version of _process_single
                from PIL import Image
                
                # Load image
                image = np.array(Image.open(image_path).convert('RGB'))
                image_float = image.astype(np.float32) / 255.0
                
                # Preprocess
                preprocessed = self.preprocessor.process(image_float)
                
                # Detect ROIs
                rois = self.detector.detect(preprocessed)
                
                # Classify ROIs
                results = []
                for roi in rois:
                    x1, y1, x2, y2, _ = roi
                    roi_img = preprocessed[y1:y2, x1:x2]
                    class_id, confidence = self.classifier.predict(roi_img)
                    
                    if class_id != -1:
                        result = {
                            'image_path': image_path,
                            'bbox': (x1, y1, x2, y2),
                            'class_id': class_id,
                            'confidence': confidence
                        }
                        results.append(result)
                        all_results.append(result)
                
                # Save visualization if needed
                if self.cfg.mode.demo.get("save_results", False):
                    output_dir = self.cfg.get("output_dir", "results/demo")
                    ensure_dir(output_dir)
                    self._save_visualization(
                        image, 
                        results, 
                        os.path.join(output_dir, f"demo_{os.path.basename(image_path)}")
                    )
            
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}")
        
        # Generate summary report
        self._generate_summary(all_results)
    
    def _visualize_results(self, image: np.ndarray, results: List[Dict], title: str = ""):
        """
        Visualize detection results.
        
        Args:
            image: Original image
            results: List of detection results
            title: Optional title for visualization
        """
        if not self.cfg.mode.demo.get("view_images", True):
            return
            
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # Color map for different classes
        cmap = plt.cm.get_cmap('tab10', 10)
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            class_id = result['class_id']
            confidence = result['confidence']
            
            # Choose color based on class_id
            color_idx = class_id % 10
            color = cmap(color_idx)
            
            # Draw rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                            edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(x1, y1-10, f"Line {class_id} ({confidence:.2f})",
                    color=color, fontsize=12, backgroundcolor='white')
        
        plt.title(f"Detection Results - {title}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _save_visualization(self, image: np.ndarray, results: List[Dict], output_path: str):
        """
        Save visualization to file.
        
        Args:
            image: Original image
            results: List of detection results
            output_path: Path to save visualization
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # Color map for different classes
        cmap = plt.cm.get_cmap('tab10', 10)
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            class_id = result['class_id']
            confidence = result['confidence']
            
            # Choose color based on class_id
            color_idx = class_id % 10
            color = cmap(color_idx)
            
            # Draw rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                            edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(x1, y1-10, f"Line {class_id} ({confidence:.2f})",
                    color=color, fontsize=12, backgroundcolor='white')
        
        plt.title(f"Detection Results")
        plt.axis('off')
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        ensure_dir(output_dir)
        
        # Save figure
        plt.savefig(output_path, dpi=200)
        plt.close()
        
        self.logger.info(f"Visualization saved to {output_path}")
    
    def _generate_summary(self, all_results: List[Dict]):
        """
        Generate summary of batch processing results.
        
        Args:
            all_results: List of all detection results
        """
        if not all_results:
            self.logger.warning("No results to summarize")
            return
            
        self.logger.info("=== Summary ===")
        self.logger.info(f"Processed {len(set([r['image_path'] for r in all_results]))} images")
        self.logger.info(f"Detected {len(all_results)} metro signs")
        
        # Count detections by class
        class_counts = {}
        for result in all_results:
            class_id = result['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        self.logger.info("Detections by class:")
        for class_id, count in sorted(class_counts.items()):
            self.logger.info(f"  Line {class_id}: {count}")
        
        # Create summary visualization if needed
        if self.cfg.mode.demo.get("save_results", False):
            output_dir = self.cfg.get("output_dir", "results/demo")
            ensure_dir(output_dir)
            
            # Create bar chart of class distribution
            plt.figure(figsize=(10, 6))
            plt.bar(
                [f"Line {class_id}" for class_id in sorted(class_counts.keys())],
                [class_counts[class_id] for class_id in sorted(class_counts.keys())],
                color='blue'
            )
            plt.title("Metro Line Detections")
            plt.xlabel("Line")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=200)
            plt.close()
            
            self.logger.info(f"Summary chart saved to {os.path.join(output_dir, 'class_distribution.png')}")


def main(cfg: DictConfig):
    """
    Main entry point for demo pipeline.
    
    Args:
        cfg: Configuration object
    """
    logger = get_logger(__name__)
    
    try:
        # Create demo pipeline
        pipeline = MetroDemoPipeline(
            cfg=cfg,
            logger=logger
        )
        
        # Run demo
        pipeline.run()
        
        logger.info("Demo pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Demo pipeline execution failed: {e}") 