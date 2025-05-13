import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from omegaconf import DictConfig
import tensorflow as tf
import keras
import keras_tuner as kt

from src.data.dataset import MetroDataset
from src.preprocessing.preprocessor import PreprocessingPipeline
from src.classification.base import get_classifier
from utils.utils import get_logger, ensure_dir, plot_training_history, save_confusion_matrix
from src.roi_detection import get_detector, optimize_color_parameters
class MetroTrainPipeline:
    """
    Paris Metro Line Recognition Training Pipeline
    
    Manages the training process including:
    1. Dataset loading and preprocessing
    2. Template creation
    3. CNN model training
    4. Model evaluation
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the training pipeline.
        
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
            'classification': "Classification configuration",
            'preprocessing': "Preprocessing configuration",
            'mode.train': "Training mode configuration"
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
            self.logger.info("Initializing training components...")
            
            # Initialize preprocessor
            self.preprocessor = PreprocessingPipeline(
                cfg=self.cfg.preprocessing
            )
            
            # Initialize classifier
            self.classifier = get_classifier(
                cfg=self.cfg.classification,
            )

            # Initialize detector
            self.detector = get_detector(
                cfg=self.cfg.roi_detection,
            )
            
            self.logger.info("Training components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run(self):
        """
        Run the training pipeline.
        
        Process includes template creation, CNN training, and evaluation.
        """
        self.logger.info("=== Starting Training Pipeline ===")
        
        # Load dataset
        train_data, val_data = self._load_datasets()
        
        # Optimize ROI detector if requested
        if self.cfg.mode.train.get("optimize_detector", True):
            self._optimize_detector(train_data)
        
        # Create templates if requested
        if self.cfg.mode.train.get("create_templates", True):
            self._create_templates(train_data)
        
        if self.cfg.mode.tuning.get("enabled", False):
            self.logger.info("Starting hyperparameter tuning...")
            best_params = self._hyperparameter_tuning(train_data, val_data)
            self.logger.info(f"Best hyperparameters found: {best_params}")
        
        # Train CNN if requested
        if self.cfg.mode.train.get("train_cnn", True):
            history = self._train_cnn(train_data, val_data)
            
            # Visualize training history
            if history:
                output_dir = os.path.join(self.cfg.get("output_dir", "results"), "plots")
                ensure_dir(output_dir)
                self._save_training_history(history, os.path.join(output_dir, "training_history.png"))
        
        # Evaluate if requested
        if self.cfg.mode.train.get("evaluate_after", True):
            self._evaluate_model(val_data)
        
        self.logger.info("=== Training Completed ===")
    
    def _load_datasets(self) -> Tuple[MetroDataset, MetroDataset]:
        """
        Load and prepare training and validation datasets.
        
        Returns:
            Tuple of (training dataset, validation dataset)
        """
        self.logger.info("Loading datasets...")
        
        # Load training dataset
        train_dataset = MetroDataset(
            cfg=self.cfg.dataset,
            mode='train',
        )
        
        # Load validation dataset
        val_dataset = MetroDataset(
            cfg=self.cfg.dataset,
            mode='val',
        )
        
        self.logger.info(f"Datasets loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        return train_dataset, val_dataset
    
    def _create_templates(self, train_dataset: MetroDataset):
        """
        Create templates from training data.
        
        Args:
            train_dataset: Training dataset
        """
        self.logger.info("Creating templates from training data...")

        X_train_raw, y_train = train_dataset.get_all()


        if len(X_train_raw) == 0:
            self.logger.warning("No training samples available for template creation")
            return
            
        processed_images = []
        for i, img in enumerate(X_train_raw):
            X_train_processed = self.preprocessor.process(img)
            processed_images.append(X_train_processed)
        
        X_train_processed = np.array(processed_images)
        
        self.classifier.train(X_train_processed, y_train)
        
        self.logger.info("Templates created successfully")
    
    def _train_cnn(self, train_dataset: MetroDataset, val_dataset: MetroDataset) -> Dict[str, Any]:
        """
        Train CNN classifier.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training history
        """
        self.logger.info("Training CNN classifier...")
        
        # Get all training samples
        X_train, y_train = train_dataset.get_all()
        X_val, y_val = val_dataset.get_all()
        
        if len(X_train) == 0:
            self.logger.warning("No training samples available for CNN training")
            return {}
            
        # Preprocess images
        processed_train = []
        for img in X_train:
            processed = self.preprocessor.process(img)
            processed_train.append(processed)
        
        processed_val = []
        for img in X_val:
            processed = self.preprocessor.process(img)
            processed_val.append(processed)
        
        processed_X_train = np.array(processed_train)
        processed_X_val = np.array(processed_val)
        
        # Train CNN
        history = self.classifier.train(processed_X_train, y_train)
        if history is None:
            history = {}
        
        # Save model
        model_path = os.path.join(self.cfg.get("output_dir", "results"), "models")
        ensure_dir(model_path)
        self.classifier.save(os.path.join(model_path, "trained_model.h5"))
        
        self.logger.info("CNN training completed")
        
        return history
    
    # Todo 优化 detector 的参数
    def _optimize_detector(self, train_dataset: MetroDataset):
        self.logger.info("Optimizing detector...")
        optimized_params = optimize_color_parameters(train_dataset)
        self.detector.update_params(optimized_params)
        if self.cfg.roi_detection.get("save_params", True):
            is_saved = self.detector.save_params(self.cfg.roi_detection.get("params_dir", "models"), optimized_params)
            if is_saved:
                self.logger.info("Detector parameters saved successfully")
            else:
                self.logger.warning("Failed to save detector parameters")
    def _hyperparameter_tuning(self,train_dataset: MetroDataset,val_dataset: MetroDataset) -> Dict[str, Any]:

        self.logger.info("=== Begin Hyperparameter Tuning (ResNet50) ===")
        max_trials = self.cfg.mode.tuning.get("max_trials", 10)
        executions_per_trial = self.cfg.mode.tuning.get("executions_per_trial", 1)
        tuning_dir = self.cfg.mode.tuning.get("directory", "models/tuning")
        project_name = self.cfg.mode.tuning.get("project_name", "resnet50_tuning")

        ensure_dir(tuning_dir)

        X_train_raw, y_train = train_dataset.get_all()
        X_val_raw,   y_val   = val_dataset.get_all()
        X_train = []
        X_val = []
        for img in X_train_raw:
            X_train.append(self.preprocessor.process(img))
        for img in X_val_raw:
            X_val.append(self.preprocessor.process(img))
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        h, w = self.cfg.preprocessing.resize_shape
        input_shape = (h, w, 3)
        num_classes = self.cfg.classification.cnn.num_classes

        def build_resnet_model(hp):
            base_trainable = hp.Boolean("base_trainable", default=False)
            unfreeze_blocks = hp.Int("unfreeze_blocks", 0, 4, step=1)
            lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
            dr = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)
            base = keras.applications.ResNet50(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape
            )
            base.trainable = base_trainable
            if base_trainable and unfreeze_blocks > 0:
                for layer in base.layers[:-unfreeze_blocks * 10]:
                    layer.trainable = False
                for layer in base.layers[-unfreeze_blocks * 10:]:
                    layer.trainable = True

            inp = keras.Input(shape=input_shape)
            x = base(inp, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dropout(dr)(x)
            out = keras.layers.Dense(num_classes, activation="softmax")(x)

            model = keras.Model(inp, out)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            return model

        tuner = kt.Hyperband(
            build_resnet_model,
            objective="val_accuracy",
            max_epochs=self.cfg.mode.train.get("epochs", 20),
            factor=3,
            directory=tuning_dir,
            project_name=project_name,
            executions_per_trial=executions_per_trial
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            epochs=self.cfg.mode.train.get("epochs", 20),
            batch_size=self.cfg.mode.train.get("batch_size", 32)
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_model = tuner.get_best_models(1)[0]

        model_path = os.path.join(tuning_dir, "best_resnet50.h5")
        best_model.save(model_path)
        self.logger.info(f"Best model saved: {model_path}")

        hp_values = best_hp.values
        hp_path = os.path.join(tuning_dir, "best_hyperparams.json")
        with open(hp_path, "w") as f:
            json.dump(hp_values, f, indent=2)
        self.logger.info(f"Best hyperparameters saved: {hp_path}")

        self.classification_cfg = self.classifier.cfg
        self.classification_cfg.cnn.learning_rate = hp_values["learning_rate"]
        self.classification_cfg.cnn.dropout_rate = hp_values["dropout_rate"]
        self.classification_cfg.cnn.base_trainable = hp_values["base_trainable"]
        self.classification_cfg.cnn.unfreeze_blocks = hp_values["unfreeze_blocks"]

        return hp_values
    def _evaluate_detector(self, val_dataset: MetroDataset):
        pass

    def _evaluate_model(self, val_dataset: MetroDataset):
        """
        Evaluate trained model on validation data.
        
        Args:
            val_dataset: Validation dataset
        """
        self.logger.info("Evaluating model...")
        
        # Get validation data
        X_val, y_val = val_dataset.get_all()
        
        if len(X_val) == 0:
            self.logger.warning("No validation samples available for evaluation")
            return
            
        # Preprocess images
        processed_val = []
        for img in X_val:
            processed = self.preprocessor.process(img)
            processed_val.append(processed)
        
        processed_X_val = np.array(processed_val)
        
        if not hasattr(self.classifier, 'evaluate'):
            self.logger.warning("Classifier does not support evaluation method")
            metrics = self._calculate_metrics_manually(processed_X_val, y_val)
        else:
            metrics = self.classifier.evaluate(processed_X_val, y_val)
        
        # Log results
        if 'test_accuracy' in metrics:
            self.logger.info(f"Validation accuracy: {metrics['test_accuracy']:.4f}")
        
        # Save confusion matrix
        if 'confusion_matrix' in metrics:
            output_dir = os.path.join(self.cfg.get("output_dir", "results"), "plots")
            ensure_dir(output_dir)
            
            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            classes = sorted(val_dataset.get_unique_classes())
            
            save_confusion_matrix(
                metrics['confusion_matrix'],
                classes,
                cm_path,
                title="Validation Confusion Matrix",
                normalize=True
            )
            
            self.logger.info(f"Confusion matrix saved to {cm_path}")
    
    def _calculate_metrics_manually(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        from sklearn.metrics import confusion_matrix, classification_report
        
        self.logger.info("Manually calculating evaluation metrics...")
        metrics = {}
        
        predictions = []
        for i in range(len(X_val)):
            try:
                class_id, _ = self.classifier.predict(X_val[i])
                predictions.append(class_id)
            except Exception as e:
                self.logger.error(f"Error predicting sample {i}: {e}")
                predictions.append(-1)  
        
        correct = sum(1 for p, gt in zip(predictions, y_val) if p == gt and p != -1)
        valid_preds = sum(1 for p in predictions if p != -1)
        total = len(y_val)
        
        accuracy = correct / total if total > 0 else 0
        
        metrics['test_accuracy'] = accuracy
        
        try:
            valid_indices = [i for i, p in enumerate(predictions) if p != -1]
            valid_predictions = [predictions[i] for i in valid_indices]
            valid_y_val = [y_val[i] for i in valid_indices]
            
            all_classes = sorted(list(set(y_val)))
            cm = confusion_matrix(valid_y_val, valid_predictions, labels=all_classes)
            report = classification_report(valid_y_val, valid_predictions, labels=all_classes, output_dict=True)
            
            metrics['confusion_matrix'] = cm
            metrics['classification_report'] = report
            
            self.logger.info(f"Manual evaluation - accuracy: {accuracy:.4f} ({correct}/{total})")
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix: {e}")
        
        return metrics
    
    def _save_training_history(self, history: Dict[str, Any], output_path: str):
        """
        Save training history visualization.
        
        Args:
            history: Training history dictionary
            output_path: Path to save the visualization
        """
        self.logger.info(f"Saving training history to {output_path}")
        
        plot_training_history(
            history,
            title="CNN Training History",
            save_path=output_path
        )


def main(cfg: DictConfig):
    """
    Main entry point for training pipeline.
    
    Args:
        cfg: Configuration object
    """
    logger = get_logger(__name__)
    
    try:
        # Create training pipeline
        pipeline = MetroTrainPipeline(
            cfg=cfg,
            logger=logger
        )
        
        # Run training
        pipeline.run()
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline execution failed: {e}")
        raise