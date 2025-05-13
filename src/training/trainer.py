import os
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List, Union
from omegaconf import DictConfig

from utils.utils import get_logger, ensure_dir
from src.data.utils import resize_image, convert_to_uint8

class ModelTrainer:
    """
    模型训练器
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        初始化训练器
        
        Args:
            cfg: 训练配置
            logger: 日志记录器
        """
        self.logger = logger or get_logger(__name__)
        self._init_config(cfg)
        
    def _init_config(self, cfg: DictConfig) -> None:
        """
        初始化配置
        
        Args:
            cfg: 训练配置
        """
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size
        self.augmentation_factor = cfg.augmentation_factor
        self.input_size = cfg.input_size
        self.num_classes = cfg.num_classes
        self.learning_rate = cfg.learning_rate
        self.optimizer = cfg.optimizer
        self.early_stopping = cfg.early_stopping
        self.patience = cfg.patience
        self.save_best_only = cfg.save_best_only
        
        self.logger.info(f"ModelTrainer initialized: epochs={self.epochs}, "
                         f"batch_size={self.batch_size}, input_size={self.input_size}")
    
    def train_model(self, classifier, train_dataset, val_dataset):
        """
        训练模型
        
        Args:
            classifier: 分类器实例
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            
        Returns:
            训练历史
        """
        try:
            # 仅当CNN分类器时才训练
            if not hasattr(classifier, 'model') or classifier.model is None:
                self.logger.warning("Classifier does not have a model, skipping training")
                return None
            
            # 确保模型目录存在
            model_dir = os.path.dirname(classifier.model_path)
            ensure_dir(model_dir)
            
            # 创建新模型
            model = self._create_model()
            
            # 准备数据
            X_train, y_train = self._prepare_data(train_dataset)
            X_val, y_val = self._prepare_data(val_dataset)
            
            # 数据增强
            if self.augmentation_factor > 1:
                X_train, y_train = self._augment_data(X_train, y_train)
            
            # 编译模型
            model.compile(
                optimizer=self._get_optimizer(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 回调函数
            callbacks = self._get_callbacks(classifier.model_path)
            
            # 训练模型
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # 保存最终模型
            if not self.save_best_only:
                model.save(classifier.model_path)
                self.logger.info(f"Final model saved to {classifier.model_path}")
            
            # 更新分类器的模型
            classifier.model = model
            
            # 绘制训练历史
            self._plot_history(history)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return None
    
    def _create_model(self):
        """
        创建CNN模型
        
        Returns:
            创建的CNN模型
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
            
            model = Sequential([
                # 第一个卷积块
                Conv2D(32, (3, 3), activation='relu', padding='same', 
                       input_shape=(self.input_size, self.input_size, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # 第二个卷积块
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # 第三个卷积块
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # 全连接层
                Flatten(),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
            
            return model
            
        except ImportError:
            self.logger.error("TensorFlow or Keras not found. Cannot create CNN model.")
            return None
    
    def _get_optimizer(self):
        """
        获取优化器
        
        Returns:
            优化器
        """
        try:
            import tensorflow as tf
            
            if self.optimizer.lower() == 'adam':
                return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            elif self.optimizer.lower() == 'sgd':
                return tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            elif self.optimizer.lower() == 'rmsprop':
                return tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            else:
                self.logger.warning(f"Unknown optimizer: {self.optimizer}, using Adam")
                return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                
        except ImportError:
            self.logger.error("TensorFlow or Keras not found.")
            return None
    
    def _get_callbacks(self, model_path):
        """
        获取回调函数
        
        Args:
            model_path: 模型保存路径
            
        Returns:
            回调函数列表
        """
        try:
            import tensorflow as tf
            
            callbacks = []
            
            # 模型检查点
            if self.save_best_only:
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
                callbacks.append(checkpoint)
            
            # 早停
            if self.early_stopping:
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    verbose=1,
                    restore_best_weights=True
                )
                callbacks.append(early_stop)
            
            # 学习率降低
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            return callbacks
            
        except ImportError:
            self.logger.error("TensorFlow or Keras not found.")
            return []
    
    def _prepare_data(self, dataset):
        """
        准备训练数据
        
        Args:
            dataset: 数据集
            
        Returns:
            (特征, 标签)
        """
        # 获取所有数据
        images, labels = dataset.get_all_data()
        
        if len(images) == 0:
            self.logger.warning("No data found")
            return np.array([]), np.array([])
        
        # 调整图像大小
        resized_images = []
        for img in images:
            resized = resize_image(img, (self.input_size, self.input_size))
            resized_images.append(resized)
        
        X = np.array(resized_images, dtype=np.float32)
        
        # 将标签转换为独热编码
        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
        
        # 注意：标签需要从1开始映射到0开始
        labels_zero_indexed = [label - 1 for label in labels]
        y = to_categorical(labels_zero_indexed, num_classes=self.num_classes)
        
        return X, y
    
    def _augment_data(self, X, y):
        """
        数据增强
        
        Args:
            X: 特征
            y: 标签
            
        Returns:
            (增强后的特征, 增强后的标签)
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            # 创建数据增强器
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            # 初始化增强后的数据
            X_augmented = X.copy()
            y_augmented = y.copy()
            
            # 生成增强数据
            for i in range(self.augmentation_factor - 1):
                for j in range(len(X)):
                    img = X[j]
                    label = y[j]
                    
                    # 添加批次维度
                    img_batch = np.expand_dims(img, axis=0)
                    
                    # 生成增强后的图像
                    for augmented_img in datagen.flow(img_batch, batch_size=1):
                        # 移除批次维度
                        augmented_img = augmented_img[0]
                        
                        # 添加到数据集
                        X_augmented = np.append(X_augmented, [augmented_img], axis=0)
                        y_augmented = np.append(y_augmented, [label], axis=0)
                        
                        break  # 只取第一个增强样本
            
            self.logger.info(f"Data augmentation: original={len(X)}, augmented={len(X_augmented)}")
            return X_augmented, y_augmented
            
        except ImportError:
            self.logger.error("TensorFlow or Keras not found. Skipping data augmentation.")
            return X, y
        except Exception as e:
            self.logger.error(f"Error during data augmentation: {e}")
            return X, y
    
    def _plot_history(self, history):
        """
        绘制训练历史
        
        Args:
            history: 训练历史
        """
        try:
            import matplotlib.pyplot as plt
            
            # 创建图形
            plt.figure(figsize=(12, 5))
            
            # 准确率
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            # 损失
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            # 保存图形
            plt.tight_layout()
            plt.savefig('results/training_history.png')
            
            self.logger.info("Training history plot saved to results/training_history.png")
            
        except ImportError:
            self.logger.error("Matplotlib not found. Cannot plot training history.")
        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}") 