from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union, Dict, Any, cast

import numpy as np
from omegaconf import DictConfig
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from utils.utils import get_logger, ensure_dir
from src.data.utils import resize_image
from src.classification.base import BaseClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from typing import Dict, Any, List, Union
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PredictionResult:
    """
    Adviced by ChatGPT
    """
    class_id: int
    confidence: float
    latency_ms: Optional[float] = None  


class ImagePreprocessor:

    def __init__(self, target_shape: Tuple[int, int], logger: logging.Logger):
        self.target_height, self.target_width = target_shape
        self.logger = logger

    def preprocess(self, image: Union[np.ndarray, None]) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None / image d'entrée est None")

        h, w = image.shape[:2]
        if (h, w) != (self.target_height, self.target_width):
            image = resize_image(image, (self.target_width, self.target_height))
            self.logger.debug(f"Resized image to {self.target_width}x{self.target_height}")

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=2)

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
            self.logger.debug("Normalized image by 255.0")

        return np.expand_dims(image, axis=0)


class ModelLoadError(Exception):
    """Raised when model fails to load / Levée lorsque le modèle échoue à se charger"""
    pass


class CNNClassifier(BaseClassifier):
    """
    CNN-based classifier for metro line pictograms.
    
    Uses a deep learning model to classify ROI images.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize CNN classifier.
        
        Args:
            cfg: Configuration object
            logger: Optional logger
        """
        super().__init__(cfg, logger)
        self.model = None
        self.model_path = os.path.join(
            self.cfg.get("model_dir", "models"),
            self.cfg.get("model_file", "cnn_model.h5")
        )
        
        # CNN hyperparameters
        self.batch_size = self.cfg.cnn.get("batch_size", 32)
        self.epochs = self.cfg.cnn.get("epochs", 50)
        self.learning_rate = self.cfg.cnn.get("learning_rate", 0.001)
        self.dropout_rate = self.cfg.cnn.get("dropout_rate", 0.3)
        
        # Initialize preprocessor
        self.preprocessor = None
        
    def _build_model(self) -> None:
        """
        Build CNN model architecture.
        """
        input_shape = self.cfg.cnn.get("input_shape", (64, 64, 3))
        num_classes = self.cfg.cnn.get("num_classes", 14)  # 巴黎地铁线路数+背景类
        base_filters = self.cfg.cnn.get("base_filters", 32)
        
        self.logger.info(f"Building CNN model with input shape {input_shape}, {num_classes} classes")
        
        model_type = self.cfg.cnn.get("model_type", "simple")
        
        if model_type == "resnet":
            self.model = self._build_resnet_model(input_shape, num_classes)
        else:  # simple CNN
            self.model = self._build_simple_cnn(input_shape, num_classes, base_filters)
            
    def _build_simple_cnn(self, input_shape, num_classes, base_filters):
        """
        Build a simple CNN architecture.
        """
        model = keras.Sequential()
        
        # First convolutional block
        model.add(keras.layers.Conv2D(base_filters, (3, 3), padding='same', input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        
        # Second convolutional block
        model.add(keras.layers.Conv2D(base_filters*2, (3, 3), padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        
        # Third convolutional block
        model.add(keras.layers.Conv2D(base_filters*4, (3, 3), padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        
        # Classification layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(self.dropout_rate))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer, # type: ignore
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def _build_resnet_model(self, input_shape, num_classes):
        """
        Build a model based on ResNet50.
        """
        base_trainable = self.cfg.cnn.get("base_trainable", False)
        
        # Load ResNet50 base
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
        # Set trainable layers
        base_model.trainable = base_trainable
        
        # Create new model
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer, # type: ignore
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def predict(self, image: np.ndarray, threshold: float = 0.7) -> Tuple[int, float]:
        if self.model is None:
            if os.path.exists(self.model_path):
                self.load(self.model_path)
            else:
                self.logger.error(f"Model not found at {self.model_path}")
                return -1, 0.0

        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image

        try:
            predictions = self.model.predict(image_batch) # type: ignore
            class_id = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][class_id])

            if confidence < threshold or class_id == self.cfg.cnn.get("num_classes", 14):
                return -1, confidence

            return class_id, confidence

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return -1, 0.0
            
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None, 
              class_weights: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        Train the CNN model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Optional validation images
            y_val: Optional validation labels
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Training history
        """
        if len(X_train) == 0:
            self.logger.error("No training data provided")
            return {}
            
        self.logger.info(f"Training CNN classifier with {len(X_train)} samples")
        
        # Build or load model
        if self.model is None:
            self._build_model()
            
        # Prepare callbacks
        callbacks = self._get_callbacks()
        
        # Prepare class weights if provided
        train_class_weights = None
        if class_weights is not None:
            train_class_weights = class_weights
            self.logger.info(f"Using class weights: {train_class_weights}")
            
        # Train model
        validation_data = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            validation_data = (X_val, y_val)
        
        history = self.model.fit( # type: ignore
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=train_class_weights
        )
        
        self.logger.info("CNN training completed")
        return history.history
        
    def _get_callbacks(self):
        """
        Get callbacks for model training.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.cfg.cnn.get("early_stopping_patience", 10), 
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=self.cfg.cnn.get("reduce_lr_patience", 5)
            )
        ]
        
        # Create directory for model if it doesn't exist
        ensure_dir(os.path.dirname(self.model_path))
        
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                self.model_path,
                save_best_only=True,
                monitor='val_accuracy'
            )
        )
        
        return callbacks
        
    def get_preprocessor(self):
        """
        Get the preprocessor used by this classifier.
        
        Returns:
            Preprocessor instance or None if not available
        """
        from src.preprocessing.cnn_preprocessor import CNNPreprocessor
        
        if hasattr(self, 'preprocessor') and self.preprocessor is not None:
            return self.preprocessor
            
        preprocessor = CNNPreprocessor(self.cfg)
        self.preprocessor = preprocessor
        return preprocessor
    
    def save(self, path: str) -> None:
        """
        Save the classifier model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            self.logger.error("No model to save")
            return
            
        ensure_dir(os.path.dirname(path))
        try:
            self.model.save(path) # type: ignore
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load(self, path: str) -> None:
        """
        Load the classifier model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            self.logger.error(f"Model file not found: {path}")
            return
            
        try:
            self.model = keras.models.load_model(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = None
    
    def visualize_training_history(self, history: Dict[str, List]) -> None:
        """
        Visualize training history.
        
        Args:
            history: Training history dictionary
        """
        if not history:
            self.logger.warning("No training history to visualize")
            return
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot training & validation accuracy
        ax1.plot(history.get('accuracy', []))
        ax1.plot(history.get('val_accuracy', []))
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper left')
        
        ax2.plot(history.get('loss', []))
        ax2.plot(history.get('val_loss', []))
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()
        


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Evaluate the model on test data, including rejection (-1) class.
        
        Args:
            X_test: Test images
            y_test: Test labels
            threshold: Confidence threshold for rejection
            
        Returns:
            Dictionary with metrics and evaluation artifacts
        """
        if self.model is None:
            self.logger.warning("Model not loaded yet")
            return {}

        try:
            model_typed = cast(keras.Model, self.model)
            scores = model_typed.evaluate(X_test, y_test, verbose=0)

            predictions = model_typed.predict(X_test)
            pred_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            import matplotlib.pyplot as plt

            plt.hist(confidences, bins=20, range=(0, 1))
            plt.title("Distribution of softmax max confidence")
            plt.xlabel("Confidence")
            plt.ylabel("Number of samples")
            plt.axvline(0.7, color='red', linestyle='--', label='Rejection threshold')
            plt.legend()
            plt.tight_layout()
            plt.show()
            # Reject if confidence < threshold
            pred_classes = np.where(confidences < threshold, -1, pred_classes)

            # Convert ground truth background label to -1
            num_classes = self.cfg.cnn.get("num_classes", 14)
            original_bg_id = num_classes - 1  # 假设背景类在最后一类
            background_class_id = self.cfg.get("background_class_id", -1)
            y_test_mapped = np.where(y_test == original_bg_id, background_class_id, y_test)

            # All labels (0~N-1 + -1)
            all_labels = list(range(num_classes - 1)) + [background_class_id]

            # Compute metrics
            cm = confusion_matrix(y_test_mapped, pred_classes, labels=all_labels)
            report = classification_report(y_test_mapped, pred_classes, labels=all_labels, output_dict=True, zero_division=0)

            # Custom metrics
            correct = (pred_classes == y_test_mapped)
            accuracy = correct.mean()

            rejected = pred_classes == -1
            total_rejected = rejected.sum()
            total = len(y_test)

            bg_mask = y_test_mapped == -1
            correct_rejection = (pred_classes[bg_mask] == -1).sum()
            bg_total = bg_mask.sum()
            bg_rejection_rate = correct_rejection / bg_total if bg_total > 0 else 0.0

            rejection_rate = total_rejected / total

            self.logger.info(f"→ Total accuracy (with rejection): {accuracy:.4f}")
            self.logger.info(f"→ Background rejection rate: {correct_rejection}/{bg_total} = {bg_rejection_rate:.2%}")
            self.logger.info(f"→ Overall rejection rate: {total_rejected}/{total} = {rejection_rate:.2%}")

            self._plot_confusion_matrix(cm, labels=all_labels)

            return {
                "test_loss": scores[0],
                "test_accuracy": scores[1],
                "confusion_matrix": cm,
                "classification_report": report,
                "total_accuracy": accuracy,
                "rejection_rate": rejection_rate,
                "background_rejection_rate": bg_rejection_rate,
            }

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {}

    def _plot_confusion_matrix(self, cm: np.ndarray, labels: List[Union[int, str]]) -> None:
        """
        Plot confusion matrix with labels.
        """
        label_names = [str(l) if l != -1 else "Rejected" for l in labels]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix with Rejection Class")
        plt.tight_layout()

        save_path = os.path.join(self.cfg.get("output_dir", "results"), "plots", "confusion_matrix.png")
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Confusion matrix saved to {save_path}")