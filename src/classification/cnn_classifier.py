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
    CNN Classifier
    
    Uses a convolutional neural network to classify metro line signs.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize CNN classifier.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        super().__init__(cfg, logger)
        
        # Get configuration
        self.model_path = cfg.cnn.get("model_path", "models/cnn_model.h5")
        self.input_shape = tuple(cfg.cnn.get("input_shape", [64, 64, 3]))
        self.num_classes = cfg.cnn.get("num_classes", 14)
        self.architecture = cfg.cnn.get("architecture", "resnet")
        self.dropout_rate = cfg.cnn.get("dropout_rate", 0.5)
        self.threshold = cfg.get("threshold", 0.5)
        
        # Other parameters
        self.model: Optional[tf.keras.Model] = None
        
        # Try to load model if it exists
        try:
            if os.path.exists(self.model_path):
                self.logger.info(f"Loading model from {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
                self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load model: {e}")
    
    def _build_model(self) -> None:
        """
        Build the CNN model architecture.
        """
        self.logger.info(f"Building CNN model with {self.architecture} architecture")
        
        try:
            if self.architecture.lower() == "resnet":
                # Use ResNet50 as base model (better for desktop/server use)
                try:
                    # Try different import paths for ResNet50 based on keras version
                    if hasattr(keras.applications, 'resnet'):
                        base_model = keras.applications.resnet.ResNet50(
                            input_shape=self.input_shape,
                            include_top=False,
                            weights='imagenet'
                        )
                    elif hasattr(keras.applications, 'ResNet50'):
                        base_model = keras.applications.ResNet50(
                            input_shape=self.input_shape,
                            include_top=False,
                            weights='imagenet'
                        )
                    else:
                        # Fallback to simpler model if ResNet50 not available
                        self.logger.warning("ResNet50 not available, falling back to simple CNN")
                        self.architecture = "simple"
                        return self._build_model()
                        
                except Exception as e:
                    self.logger.warning(f"Error loading ResNet50: {e}, falling back to simple CNN")
                    self.architecture = "simple"
                    return self._build_model()
                
                # Freeze base model layers
                base_model.trainable = False
                
                # Add classification head
                inputs = tf.keras.layers.Input(shape=self.input_shape)
                x = base_model(inputs, training=False)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
                outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
                
                self.model = tf.keras.models.Model(inputs, outputs)
                
            else:
                # Simple custom CNN
                model = tf.keras.models.Sequential()
                
                # Convolutional blocks
                model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
                model.add(tf.keras.layers.MaxPooling2D((2, 2)))
                
                model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(tf.keras.layers.MaxPooling2D((2, 2)))
                
                model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(tf.keras.layers.MaxPooling2D((2, 2)))
                
                # Classification head
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(256, activation='relu'))
                model.add(tf.keras.layers.Dropout(self.dropout_rate))
                model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
                
                self.model = model
            
            # Compile model
            if self.model is not None:
                self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                self.logger.info(f"Model built with {self.model.count_params()} parameters")
            
        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            raise
    
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict class for an image.
        
        Args:
            image: Image to classify (preprocessed ROI)
            
        Returns:
            Tuple of (class_id, confidence)
        """
        if self.model is None:
            self.logger.warning("Model not loaded yet")
            return -1, 0.0
        
        try:
            # Ensure image has right dimensions and type
            if len(image.shape) == 2:  # Grayscale image
                image = np.stack([image] * 3, axis=-1)
            
            # Ensure image has right shape
            if image.shape[:2] != self.input_shape[:2]:
                from cv2 import resize
                image = resize(image, (self.input_shape[1], self.input_shape[0]))
            
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # 使用正确的类型注解进行调用
            # Make prediction
            model_typed = cast(tf.keras.Model, self.model)  # 确保类型正确
            predictions = model_typed.predict(image)
            
            # Get highest confidence class
            class_id = np.argmax(predictions[0])
            confidence = predictions[0][class_id]
            
            # Apply threshold
            if confidence < self.threshold:
                return -1, float(confidence)
            
            return int(class_id), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return -1, 0.0
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the classifier with provided data.
        
        Args:
            X_train: Training images
            y_train: Training labels
            
        Returns:
            Dictionary with training history
        """
        self.logger.info(f"Training CNN classifier with {len(X_train)} samples")
        
        # Build model if not already built
        if self.model is None:
            self._build_model()
        
        # 初始化一个空的历史记录，以防后续代码提前返回
        empty_history: Dict[str, Any] = {}
        
        # Create data generators for augmentation
        if self.cfg.cnn.get("use_augmentation", True):
            try:
                # 尝试不同的导入路径
                if hasattr(tf.keras.preprocessing.image, 'ImageDataGenerator'):
                    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        horizontal_flip=False,
                        zoom_range=0.1,
                        brightness_range=[0.9, 1.1],
                        validation_split=0.2  # Use 20% of data for validation
                    )
                else:
                    # 备用方案：使用keras_cv或直接tf.image进行数据增强
                    self.logger.warning("ImageDataGenerator not available, disabling augmentation")
                    return self._train_without_augmentation(X_train, y_train)
                    
                # Split data into training and validation
                val_split = int(len(X_train) * 0.8)
                X_val = X_train[val_split:]
                y_val = y_train[val_split:]
                X_train = X_train[:val_split]
                y_train = y_train[:val_split]
                
                # Create train and validation generators
                train_generator = datagen.flow(
                    X_train, y_train,
                    batch_size=self.cfg.cnn.get("batch_size", 32)
                )
                
                validation_generator = datagen.flow(
                    X_val, y_val,
                    batch_size=self.cfg.cnn.get("batch_size", 32)
                )
                
                # Setup callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
                ]
                
                # Create directory for model if it doesn't exist
                ensure_dir(os.path.dirname(self.model_path))
                
                callbacks.append(
                    tf.keras.callbacks.ModelCheckpoint(
                        self.model_path,
                        save_best_only=True,
                        monitor='val_accuracy'
                    )
                )
                
                # Train model
                if self.model is not None:
                    model_typed = cast(tf.keras.Model, self.model)  # 确保类型正确
                    history = model_typed.fit(
                        train_generator,
                        epochs=self.cfg.cnn.get("epochs", 50),
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        verbose=1  # 整数类型
                    )
                    
                    self.logger.info("Training completed")
                    return history.history
                else:
                    self.logger.error("Model is not initialized, cannot train")
                    return empty_history
            except Exception as e:
                self.logger.error(f"Error training with augmentation: {e}, falling back to simple training")
                return self._train_without_augmentation(X_train, y_train)
            
        else:
            return self._train_without_augmentation(X_train, y_train)
    
    def _train_without_augmentation(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train model without using data augmentation.
        
        Args:
            X_train: Training images
            y_train: Training labels
            
        Returns:
            Dictionary with training history
        """
        # 初始化一个空的历史记录，以防后续代码返回None
        empty_history: Dict[str, Any] = {}
        
        # Without augmentation, use simple train/val split
        val_split = int(len(X_train) * 0.8)
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Create directory for model if it doesn't exist
        ensure_dir(os.path.dirname(self.model_path))
        
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                self.model_path,
                save_best_only=True,
                monitor='val_accuracy'
            )
        )
        
        # Train model
        if self.model is not None:
            model_typed = cast(tf.keras.Model, self.model)  # 确保类型正确
            history = model_typed.fit(
                X_train, y_train,
                epochs=self.cfg.cnn.get("epochs", 50),
                batch_size=self.cfg.cnn.get("batch_size", 32),
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1  # 整数类型
            )
            
            return history.history
        else:
            self.logger.error("Model is not initialized, cannot train")
            return empty_history
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            self.logger.warning("No model to save")
            return
            
        try:
            # Create directory if it doesn't exist
            ensure_dir(os.path.dirname(path))
            
            # Save model
            model_typed = cast(tf.keras.Model, self.model)  # 确保类型正确
            model_typed.save(path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        try:
            if not os.path.exists(path):
                self.logger.warning(f"Model file does not exist: {path}")
                return
                
            self.model = tf.keras.models.load_model(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
    
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
        
        # Plot training & validation loss
        ax2.plot(history.get('loss', []))
        ax2.plot(history.get('val_loss', []))
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            self.logger.warning("Model not loaded yet")
            return {}
            
        try:
            # Evaluate model
            model_typed = cast(tf.keras.Model, self.model)  # 确保类型正确
            scores = model_typed.evaluate(X_test, y_test)
            
            metrics = {
                'test_loss': scores[0],
                'test_accuracy': scores[1]
            }
            
            # Calculate confusion matrix
            model_typed = cast(tf.keras.Model, self.model)  # 确保类型正确
            predictions = model_typed.predict(X_test)
            pred_classes = np.argmax(predictions, axis=1)
            
            from sklearn.metrics import confusion_matrix, classification_report
            
            cm = confusion_matrix(y_test, pred_classes)
            report = classification_report(y_test, pred_classes, output_dict=True)
            
            metrics['confusion_matrix'] = cm
            metrics['classification_report'] = report
            
            self.logger.info(f"Evaluation results: loss={metrics['test_loss']:.4f}, accuracy={metrics['test_accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {}
