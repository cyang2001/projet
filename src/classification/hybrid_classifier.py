import logging
from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np
from omegaconf import DictConfig

from src.classification.base import BaseClassifier
from src.classification.template_classifier import TemplateClassifier
from src.classification.cnn_classifier import CNNClassifier
from utils.utils import get_logger

class HybridClassifier(BaseClassifier):
    """
    Hybrid Classifier
    
    Combines template matching and CNN classification for more robust results.
    Can use either method as primary classifier with the other as a fallback.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize hybrid classifier.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        super().__init__(cfg, logger)
        
        # Configuration
        self.primary = cfg.hybrid.get("primary", "cnn")
        self.fallback_threshold = cfg.hybrid.get("fallback_threshold", 0.6)
        self.threshold = cfg.get("threshold", 0.5)
        
        # Initialize component classifiers
        self.logger.info(f"Initializing hybrid classifier with primary: {self.primary}")
        
        # Template classifier
        self.template_classifier = TemplateClassifier(cfg, logger)
        
        # CNN classifier
        self.cnn_classifier = CNNClassifier(cfg, logger)
    
    def predict(self, image: np.ndarray) -> Tuple[int, Union[float, np.floating[Any]]]:
        """
        Predict class for an image.
        
        Uses primary classifier first, then falls back to secondary if confidence is low.
        
        Args:
            image: Image to classify (preprocessed ROI)
            
        Returns:
            Tuple of (class_id, confidence)
        """
        # Determine primary and secondary classifiers
        primary_classifier = self.cnn_classifier if self.primary == "cnn" else self.template_classifier
        secondary_classifier = self.template_classifier if self.primary == "cnn" else self.cnn_classifier
        
        # Get primary prediction
        class_id, confidence = primary_classifier.predict(image)
        
        # If confidence is high enough, return the result
        if confidence >= self.fallback_threshold:
            self.logger.debug(f"Primary classifier ({self.primary}) confidence: {confidence:.4f}")
            return class_id, confidence
        
        # Otherwise, try the secondary classifier
        self.logger.debug(f"Primary classifier ({self.primary}) confidence too low: {confidence:.4f}, using fallback")
        secondary_class_id, secondary_confidence = secondary_classifier.predict(image)
        
        # If secondary has higher confidence, use it
        if secondary_confidence > confidence:
            self.logger.debug(f"Secondary classifier confidence: {secondary_confidence:.4f}")
            return secondary_class_id, secondary_confidence
        
        # Otherwise, still use primary but with original lower confidence
        return class_id, confidence
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train both component classifiers.
        
        Args:
            X_train: Training images
            y_train: Training labels
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Training hybrid classifier (template + CNN)")
        
        # Train template classifier
        self.logger.info("Training template classifier...")
        self.template_classifier.train(X_train, y_train)
        
        # Train CNN classifier
        self.logger.info("Training CNN classifier...")
        history = self.cnn_classifier.train(X_train, y_train)
        
        self.logger.info("Hybrid classifier training completed")
        
        return {
            "cnn_history": history
        }
    
    def save(self, path: str) -> None:
        """
        Save both component classifiers.
        
        Args:
            path: Base path for saving models
        """
        # Save CNN model
        cnn_path = f"{path}_cnn.h5"
        self.cnn_classifier.save(cnn_path)
        
        # Template models are saved during training
        self.logger.info(f"Hybrid classifier models saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load both component classifiers.
        
        Args:
            path: Base path for loading models
        """
        # Load CNN model
        cnn_path = f"{path}_cnn.h5"
        self.cnn_classifier.load(cnn_path)
        
        # Load template models
        template_path = self.template_classifier.template_dir
        self.template_classifier.load(template_path)
        
        self.logger.info(f"Hybrid classifier models loaded from {path}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the hybrid classifier on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        confidences = []
        
        # Get predictions for each test image
        for i, image in enumerate(X_test):
            class_id, confidence = self.predict(image)
            predictions.append(class_id)
            confidences.append(confidence)
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
        accuracy = correct / len(y_test) if len(y_test) > 0 else 0
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        try:
            cm = confusion_matrix(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)
            
            self.logger.info(f"Hybrid classifier evaluation: accuracy={accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': predictions,
                'confidences': confidences
            }
        
        except Exception as e:
            self.logger.error(f"Error generating evaluation metrics: {e}")
            
            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'confidences': confidences
            } 