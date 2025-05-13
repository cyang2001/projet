import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import scipy.io as sio
from typing import List, Optional, Tuple, Dict, Any, Union
from omegaconf import DictConfig

from src.data.dataset import MetroDataset
from src.preprocessing.preprocessor import PreprocessingPipeline
from src.roi_detection import get_detector, visualize_detection_steps 
from src.classification.base import get_classifier
from utils.utils import get_logger, ensure_dir, save_confusion_matrix, visualize_detection

class MetroTestPipeline:
    """
    Paris Metro Line Recognition Test Pipeline
    
    Processes test images to detect and classify metro line signs, and evaluates accuracy.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the test pipeline.
        
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
            'dataset': "Dataset configuration",
            'roi_detection': "ROI detector configuration",
            'classification': "Classification configuration",
            'preprocessing': "Preprocessing configuration",
            'mode.test': "Testing mode configuration"
        }
        
        for key, description in required_configs.items():
            parts = key.split('.')
            curr = self.cfg
            for part in parts:
                if not hasattr(curr, part):
                    raise ValueError(f"Configuration missing: {key} ({description})")
                curr = getattr(curr, part)
        
        self.logger.info("Configuration validation passed")
    
    def _init_components(self):
        """
        Initialize pipeline components.
        """
        try:
            self.logger.info("Initializing test components...")
            
            self.preprocessor = PreprocessingPipeline(
                cfg=self.cfg.preprocessing
            )
            
            self.detector = get_detector(
                cfg=self.cfg.roi_detection,
            )
            
            self.classifier = get_classifier(
                cfg=self.cfg.classification,
            )
            
            self.logger.info("Test components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run(self):
        """
        Run the test pipeline.
        
        Process test images, detect ROIs, classify them, and evaluate results.
        """
        self.logger.info("=== Starting Test Pipeline ===")
        
        test_dataset = MetroDataset(
            cfg=self.cfg.dataset,
            mode='test',
        )
        
        self.logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
        
        results = []
        predictions = []
        ground_truth = []
        
        for idx in range(len(test_dataset)):
            image, annotations = test_dataset.get_image_with_annotations(idx)
            image_id = test_dataset.df.iloc[idx]['image_id']
            self.logger.info(f"Processing image ID: {image_id}")
            visualize_detection_steps(self.detector, image)
            processed_image = self.preprocessor.process(image)
            
            detected_rois = self.detector.detect(processed_image)
            
            self.logger.info(f"Detected {len(detected_rois)} ROIs")
            
            detected_classes = []
            for roi in detected_rois:
                x1, y1, x2, y2, _ = roi
                
                roi_img = processed_image[y1:y2, x1:x2]
                
                if roi_img.size == 0 or roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
                    self.logger.warning(f"Invalid ROI: {roi}")
                    continue
                
                try:
                    class_id, confidence = self.classifier.predict(roi_img)
                    
                    if class_id != -1:  
                        detected_classes.append((class_id, (x1, y1, x2, y2), confidence))
                except Exception as e:
                    self.logger.error(f"Error classifying ROI: {e}")
            
            gt_classes = [(ann[4], ann[:4]) for ann in annotations]  
            
            image_result = {
                'image_id': int(image_id),
                'detected': [(cls, bbox, conf) for cls, bbox, conf in detected_classes],
                'ground_truth': gt_classes
            }
            results.append(image_result)
            
            for cls, _, _ in detected_classes:
                predictions.append(cls)
            for cls, _ in gt_classes:
                ground_truth.append(cls)
            
            if self.cfg.mode.test.get("view_images", False):
                self._visualize_results(image, detected_classes, gt_classes, image_id)
        
        self._calculate_metrics(results, predictions, ground_truth)
        
        self._save_results(results)
        
        self.logger.info("=== Test Completed ===")
    
    def _visualize_results(self, image: np.ndarray, detections: List[Tuple], 
                          ground_truth: List[Tuple], image_id: int):
        """
        Visualize detection and classification results.
        
        Args:
            image: Original image
            detections: List of (class_id, bbox, confidence)
            ground_truth: List of (class_id, bbox)
            image_id: Image identifier
        """
        det_boxes = [(box[0], box[1], box[2], box[3], cls) for cls, box, _ in detections]
        
        gt_boxes = [(box[0], box[1], box[2], box[3], cls) for cls, box in ground_truth]
        
        if self.cfg.mode.test.get("save_visualization", False):
            output_dir = os.path.join(self.cfg.get("output_dir", "results"), "visualizations")
            ensure_dir(output_dir)
            save_path = os.path.join(output_dir, f"test_result_{image_id}.png")
        else:
            save_path = None
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Detections")
        
        for x1, y1, x2, y2, cls in det_boxes:
            rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, f"Line {cls}", color='red', fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title("Ground Truth")
        
        for x1, y1, x2, y2, cls in gt_boxes:
            rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, f"Line {cls}", color='green', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            self.logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def _calculate_metrics(self, results: List[Dict], predictions: List[int], 
                          ground_truth: List[int]):
        """
        Calculate evaluation metrics.
        
        Args:
            results: List of detection results per image
            predictions: List of predicted classes
            ground_truth: List of ground truth classes
        """
        self.logger.info("Calculating evaluation metrics...")
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        try:
            if predictions and ground_truth:
                all_classes = sorted(list(set(predictions + ground_truth)))
                
                cm = confusion_matrix(ground_truth, predictions, labels=all_classes)
                report = classification_report(ground_truth, predictions, 
                                             labels=all_classes, output_dict=True)
                
                self.logger.info(f"Classification report:\n{classification_report(ground_truth, predictions, labels=all_classes)}")
                
                output_dir = os.path.join(self.cfg.get("output_dir", "results"), "plots")
                ensure_dir(output_dir)
                
                save_confusion_matrix(
                    cm,
                    all_classes,
                    os.path.join(output_dir, "test_confusion_matrix.png"),
                    title="Test Confusion Matrix",
                    normalize=True
                )
                
                correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
                total = len(ground_truth)
                accuracy = correct / total if total > 0 else 0
                
                self.logger.info(f"Test accuracy: {accuracy:.4f} ({correct}/{total})")
                
                metrics = {
                    'confusion_matrix': cm,
                    'classification_report': report,
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                }
                
                output_dir = self.cfg.get("output_dir", "results")
                ensure_dir(output_dir)
                
                np.savez(
                    os.path.join(output_dir, "test_metrics.npz"),
                    **metrics
                )
                
                self.logger.info(f"Evaluation metrics saved to {output_dir}/test_metrics.npz")
            else:
                self.logger.warning("No predictions or ground truth available for evaluation")
        
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
    
    def _save_results(self, results: List[Dict]):
        """
        Save test results.
        
        Args:
            results: List of detection results per image
        """
        if not results:
            self.logger.warning("No results to save")
            return
            
        try:
            if self.cfg.mode.test.get("save_mat", True):
                output_dir = self.cfg.get("output_dir", "results")
                ensure_dir(output_dir)
                
                mat_results = []
                
                for result in results:
                    image_id = result['image_id']
                    
                    for cls, bbox, conf in result['detected']:
                        x1, y1, x2, y2 = bbox
                        
                        mat_results.append([image_id, y1, y2, x1, x2, cls])
                
                from src.data.utils import save_results
                mat_path = os.path.join(output_dir, "test_results.mat")
                save_results(mat_results, mat_path)
                
                self.logger.info(f"Results saved to {mat_path}")
                
            output_dir = self.cfg.get("output_dir", "results")
            ensure_dir(output_dir)
            
            serialized_results = {
                f"result_{i}": str(result) for i, result in enumerate(results)
            }
            
            np.savez(
                os.path.join(output_dir, "test_results.npz"),
                **serialized_results
            )
            
            self.logger.info(f"Detailed results saved to {output_dir}/test_results.npz")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


def main(cfg: DictConfig):
    """
    Main entry point for test pipeline.
    
    Args:
        cfg: Configuration object
    """
    logger = get_logger(__name__)
    
    try:
        # Create test pipeline
        pipeline = MetroTestPipeline(
            cfg=cfg,
        )
        
        # Run testing
        pipeline.run()
        
        logger.info("Test pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Test pipeline execution failed: {e}")
        raise 