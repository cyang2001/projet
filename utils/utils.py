import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Union, Optional
from omegaconf import DictConfig
import glob
from pathlib import Path
from matplotlib.patches import Rectangle
import cv2

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with specified name and level.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure logger if it hasn't been configured yet
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handler
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger

def ensure_dir(path: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    """
    try:
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error creating directory {path}: {e}")

def find_file(filename: str, file_dir: str) -> Optional[str]:
    logger = get_logger(__name__)
    
    if os.path.exists(os.path.join(file_dir, filename)):
        return os.path.join(file_dir, filename)
    else:
        return None

def draw_rectangle(img: np.ndarray, bbox: Tuple[int, int, int, int], label: Optional[str] = None, 
                  color: Tuple[float, float, float] = (1.0, 0, 0), thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制矩形
    
    Args:
        img: 输入图像
        bbox: 边界框坐标 (x1, y1, x2, y2)
        label: 标签文本
        color: 矩形颜色 (R, G, B)
        thickness: 线条粗细
        
    Returns:
        绘制了矩形的图像
    """
    # 确保图像是uint8类型
    if img.dtype != np.uint8:
        # 如果是浮点型且范围在[0, 1]
        if img.dtype in [np.float32, np.float64] and np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # 拷贝图像，避免修改原图
    img_copy = img.copy()
    
    # 转换颜色从[0,1]到[0,255]
    color_bgr = [int(c * 255) for c in color[::-1]]  # RGB到BGR并缩放到[0,255]
    
    # 绘制矩形
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color_bgr, thickness)
    
    # 如果有标签，添加文本
    if label:
        cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color_bgr, 1, cv2.LINE_AA)
    
    return img_copy

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], 
                         normalize: bool = False, 
                         title: str = 'Confusion matrix',
                         save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
    
    plt.show()

def save_results_to_mat(results: List[List[Any]], output_file: str = 'myResults.mat') -> str:

    import scipy.io as sio
    import numpy as np
    results_array = np.array(results, dtype=np.float64)
    
    sio.savemat(output_file, {'BD': results_array})
    
    return output_file

def save_confusion_matrix(
    cm: np.ndarray,
    classes: List[int],
    output_path: str,
    title: str = 'Confusion Matrix',
    normalize: bool = False
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: Class labels
        output_path: Path to save the visualization
        title: Plot title
        normalize: Whether to normalize values
    """
    if normalize:
        row_sums = cm.sum(axis=1)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        cm = cm.astype('float') / row_sums[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=100)
    plt.close()

def visualize_detection(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int, int]],
    title: str = "Detection Results",
    show: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize detection results.
    
    Args:
        image: Image array
        boxes: List of bounding boxes as (x1, y1, x2, y2, class_id)
        title: Plot title
        show: Whether to show the plot
        save_path: Path to save the visualization
    """
    # Convert to uint8 if float
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Create a copy of the image
    vis_image = image.copy()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display image
    plt.imshow(vis_image)
    
    # Color map for different classes
    cmap = plt.cm.get_cmap('tab10', 10)
    
    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2, class_id = box
        
        # Choose color based on class ID
        color_idx = class_id % 10
        color = cmap(color_idx)
        
        # Draw rectangle
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        
        # Draw label
        plt.text(x1, y1 - 5, f"Class {class_id}",
                color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=100)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_results(
    images: List[np.ndarray],
    true_labels: List[int],
    pred_labels: List[int],
    confidences: List[float],
    max_images: int = 20,
    title: str = "Classification Results",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize classification results.
    
    Args:
        images: List of image arrays
        true_labels: List of true class labels
        pred_labels: List of predicted class labels
        confidences: List of prediction confidences
        max_images: Maximum number of images to visualize
        title: Main title of the plot
        save_path: Path to save the visualization
    """
    n_images = min(len(images), max_images)
    if n_images <= 0:
        return
    
    # Calculate grid size
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create figure
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    plt.suptitle(title, fontsize=16)
    
    # Plot images
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Convert to uint8 if needed
        img = images[i]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        
        plt.imshow(img)
        
        # Determine color based on correct/incorrect prediction
        color = 'green' if pred_labels[i] == true_labels[i] else 'red'
        
        # Add prediction label
        plt.title(f"True: {true_labels[i]}, Pred: {pred_labels[i]}\nConf: {confidences[i]:.2f}", 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for main title
    
    # Save if path provided
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=100)
    
    plt.show()

def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training metrics
        title: Plot title
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=16)
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='Train')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    if 'loss' in history:
        plt.plot(history['loss'], label='Train')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for main title
    
    # Save if path provided
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=100)
    
    plt.show()

def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio and padding if necessary.
    
    Args:
        image: Input image
        target_size: Target size as (height, width)
        pad_color: Padding color as (R, G, B)
        
    Returns:
        Resized and padded image
    """
    # Get original dimensions
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_h / h, target_w / w)
    
    # Calculate new dimensions
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create target image with padding
    if len(image.shape) == 3:
        # Color image
        result = np.ones((target_h, target_w, 3), dtype=image.dtype) * np.array(pad_color, dtype=image.dtype)
        # Convert pad_color if image is float
        if image.dtype == np.float32 or image.dtype == np.float64:
            result = result / 255.0
    else:
        # Grayscale image
        result = np.ones((target_h, target_w), dtype=image.dtype) * pad_color[0]
        if image.dtype == np.float32 or image.dtype == np.float64:
            result = result / 255.0
    
    # Calculate padding
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place the resized image in the center
    if len(image.shape) == 3:
        result[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
    else:
        result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return result 