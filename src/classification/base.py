import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any, Dict, List, Optional, Union
from omegaconf import DictConfig

from utils.utils import get_logger

class BaseClassifier(ABC):
    """
    Base Classifier interface
    
    All classifiers must implement the predict method.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize base classifier.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
        
    @abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict class for an image.
        
        Args:
            image: Image to classify (preprocessed ROI)
            
        Returns:
            Tuple of (class_id, confidence)
        """
        pass
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              class_weights: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        Train the classifier with provided data.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Optional validation images
            y_val: Optional validation labels
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Dictionary with training history or metrics
        """
        pass
    
    def get_preprocessor(self):
        """
        Get the preprocessor used by this classifier.
        
        Returns:
            Preprocessor instance or None if not available
        """
        return None
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.warning("Evaluation not implemented for this classifier. Using manual evaluation.")
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        metrics = {}
        
        # Manually predict each sample
        predictions = []
        for i in range(len(X_test)):
            try:
                class_id, _ = self.predict(X_test[i])
                predictions.append(class_id)
            except Exception as e:
                self.logger.error(f"Error predicting sample {i}: {e}")
                predictions.append(-1)  # Use -1 to indicate prediction failure
        
        # Calculate accuracy
        correct = sum(1 for p, gt in zip(predictions, y_test) if p == gt and p != -1)
        valid_preds = sum(1 for p in predictions if p != -1)
        total = len(y_test)
        
        accuracy = correct / total if total > 0 else 0
        
        metrics['test_accuracy'] = accuracy
        
        # Calculate confusion matrix
        try:
            valid_indices = [i for i, p in enumerate(predictions) if p != -1]
            valid_predictions = [predictions[i] for i in valid_indices]
            valid_y_test = [y_test[i] for i in valid_indices]
            
            all_classes = sorted(list(set(y_test)))
            cm = confusion_matrix(valid_y_test, valid_predictions, labels=all_classes)
            report = classification_report(valid_y_test, valid_predictions, labels=all_classes, output_dict=True)
            
            metrics['confusion_matrix'] = cm
            metrics['classification_report'] = report
            
            self.logger.info(f"Manual evaluation - accuracy: {accuracy:.4f} ({correct}/{total})")
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix: {e}")
        
        return metrics
        
    def save(self, path: str) -> None:
        """
        Save the classifier model to disk.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError("Saving not implemented for this classifier")
        
    def load(self, path: str) -> None:
        """
        Load the classifier model from disk.
        
        Args:
            path: Path to load the model from
        """
        raise NotImplementedError("Loading not implemented for this classifier")


def get_classifier(cfg: DictConfig, logger: Optional[logging.Logger] = None) -> BaseClassifier:
    """
    Factory function to get a classifier based on the configuration.
    
    Args:
        cfg: Configuration dictionary
        logger: Optional logger
        
    Returns:
        Initialized classifier object
        
    Raises:
        ValueError: If classifier type is not supported
    """
    classifier_name = cfg.get("name", "template_classifier")
    
    if logger:
        logger.info(f"Creating classifier of type: {classifier_name}")
    else:
        logger = get_logger(__name__)
        logger.info(f"Creating classifier of type: {classifier_name}")
    if classifier_name == "template_classifier":
        from src.classification.template_classifier import TemplateClassifier
        return TemplateClassifier(cfg, logger)
    elif classifier_name == "cnn_classifier":
        from src.classification.cnn_classifier import CNNClassifier
        return CNNClassifier(cfg, logger)
    elif classifier_name == "hybrid_classifier":
        from src.classification.hybrid_classifier import HybridClassifier
        return HybridClassifier(cfg, logger)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_name}") 