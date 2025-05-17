import os
import numpy as np
import scipy.io as sio
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from utils.utils import get_logger

def load_image(image_path: str, normalize: bool = True) -> Optional[np.ndarray]:
    """
    加载图像
    
    Args:
        image_path: 图像文件路径
        normalize: 是否归一化到[0,1]范围
        
    Returns:
        加载的图像数组，如果失败则返回None
    """
    try:
        # 确保文件存在
        if not os.path.exists(image_path):
            return None
        
        # 使用PIL加载图像并转换为RGB
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # 如果需要归一化
        if normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to load image {image_path}: {e}")
        return None

def save_results(results: List[List[Any]], output_file: str = 'myResults.mat') -> str:
    """
    将结果保存为.mat文件，格式为(image_id, y1, y2, x1, x2, class_id)
    
    Args:
        results: 结果列表，格式为[[image_id, class_id, confidence, x1, y1, x2, y2], ...]
        output_file: 输出文件路径
        
    Returns:
        保存的文件路径
    """
    # 确保结果不为空
    if not results:
        logger = get_logger(__name__)
        logger.warning("No results to save.")
        return output_file
    
    # 将结果转换为正确的格式 (image_id, y1, y2, x1, x2, class_id)
    reformatted_results = []
    for result in results:
        if len(result) >= 7:  # 确保有足够的元素
            image_id, class_id, confidence, x1, y1, x2, y2 = result[:7]
            # 转换为要求的格式
            reformatted_results.append([image_id, y1, y2, x1, x2, class_id])
    
    # 将结果转换为numpy数组
    results_array = np.array(reformatted_results, dtype=np.float64)
    
    # 保存为mat文件
    try:
        logger = get_logger(__name__)
        logger.info(f"Saving {len(reformatted_results)} results to {output_file} with format (image_id, y1, y2, x1, x2, class_id)")
        sio.savemat(output_file, {'BD': results_array})
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to save results to {output_file}: {e}")
    
    return output_file

def resize_image(image: np.ndarray, target_size: Tuple[int, int], keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize an image to target size, optionally maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target size as (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if image is None:
        return np.zeros((*target_size, 3) if len(image.shape) == 3 else target_size, dtype=np.uint8)
        
    target_width, target_height = target_size
    
    if keep_aspect_ratio:
        # Calculate scale to maintain aspect ratio
        h, w = image.shape[:2]
        scale = min(target_height / h, target_width / w)
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create target image with padding
        if len(image.shape) == 3:
            # Color image
            result = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            # Grayscale image
            result = np.zeros((target_height, target_width), dtype=image.dtype)
        
        # Calculate position to place the resized image
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        # Place the resized image
        if len(image.shape) == 3:
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
        else:
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
        return result
    else:
        # Direct resize
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def normalize_image(image: np.ndarray, mode: str = 'minmax') -> np.ndarray:
    """
    Normalize image values.
    
    Args:
        image: Input image
        mode: Normalization mode ('minmax', 'standard', 'mean')
        
    Returns:
        Normalized image
    """
    if mode == 'minmax':
        # Scale to [0, 1]
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image
    elif mode == 'standard':
        # Standardize to mean=0, std=1
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        return image - mean
    elif mode == 'mean':
        # Just subtract mean
        return image - np.mean(image)
    else:
        return image

def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 format.
    
    Args:
        image: Input image (float or int)
        
    Returns:
        Image in uint8 format
    """
    if image.dtype == np.uint8:
        return image
        
    if image.dtype == np.float32 or image.dtype == np.float64:
        if np.max(image) <= 1.0:
            image = image * 255.0
    
    return np.clip(image, 0, 255).astype(np.uint8)

def show_image(image: np.ndarray, title: str = "Image", wait: bool = True) -> None:
    """
    显示图像
    
    Args:
        image: 图像数组
        title: 窗口标题
        wait: 是否等待键盘输入
    """
    # 确保图像是uint8类型
    img_display = convert_to_uint8(image)
    
    # 处理BGR到RGB转换
    if len(img_display.shape) == 3 and img_display.shape[2] == 3:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    # 显示图像
    plt.figure(figsize=(10, 8))
    plt.imshow(img_display)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    if wait:
        plt.pause(0.1)  # 暂停一小段时间以显示图像

def show_image_with_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int, int]], 
                          class_names: Optional[Dict[int, str]] = None, title: str = "Detection Results",
                          wait: bool = True) -> None:
    import matplotlib.patches as patches
    
    # 确保图像是RGB格式
    img_display = convert_to_uint8(image)
    if len(img_display.shape) == 3 and img_display.shape[2] == 3:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    # 创建图像
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_display)
    
    # 颜色映射
    cmap = plt.cm.get_cmap('tab10', 10)
    
    # 绘制边界框
    for i, (x1, y1, x2, y2, class_id) in enumerate(boxes):
        # 根据类别ID选择颜色
        color_idx = (class_id % 10)
        color = cmap(color_idx)
        
        # 创建矩形
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 添加标签
        label = f"Class {class_id}"
        if class_names and class_id in class_names:
            label = class_names[class_id]
        
        ax.text(x1, y1-10, label, color=color, fontsize=10, 
                backgroundcolor='white')
    
    # 设置标题和轴
    ax.set_title(title)
    ax.axis('off')
    
    # 显示图像
    plt.tight_layout()
    plt.show()
    
    if wait:
        plt.pause(0.1) 

def crop_roi(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a region of interest from an image.
    
    Args:
        image: Input image
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure valid bbox
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    # Crop region
    return image[y1:y2, x1:x2].copy()

def pad_image(image: np.ndarray, pad_size: int, pad_value: int = 0) -> np.ndarray:
    """
    Add padding around an image.
    
    Args:
        image: Input image
        pad_size: Size of padding in pixels
        pad_value: Value to use for padding
        
    Returns:
        Padded image
    """
    if len(image.shape) == 3:
        # Color image
        h, w, c = image.shape
        padded = np.ones((h + 2 * pad_size, w + 2 * pad_size, c), dtype=image.dtype) * pad_value
        padded[pad_size:pad_size+h, pad_size:pad_size+w] = image
    else:
        # Grayscale image
        h, w = image.shape
        padded = np.ones((h + 2 * pad_size, w + 2 * pad_size), dtype=image.dtype) * pad_value
        padded[pad_size:pad_size+h, pad_size:pad_size+w] = image
        
    return padded

def augment_image(image: np.ndarray, angle: float = 0, flip_h: bool = False, flip_v: bool = False,
                 scale: float = 1.0, brightness: float = 0.0) -> np.ndarray:
    """
    Apply augmentation to an image.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        flip_h: Whether to flip horizontally
        flip_v: Whether to flip vertically
        scale: Scaling factor
        brightness: Brightness adjustment
        
    Returns:
        Augmented image
    """
    # Convert to uint8 if needed
    original_dtype = image.dtype
    if original_dtype == np.float32 or original_dtype == np.float64:
        img = convert_to_uint8(image)
    else:
        img = image.copy()
    
    h, w = img.shape[:2]
    
    # Apply rotation
    if angle != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Apply scaling
    if scale != 1.0:
        scaled_h, scaled_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (scaled_w, scaled_h))
        
        # Pad or crop to original size
        if scale > 1.0:
            # Need to crop
            start_h = (scaled_h - h) // 2
            start_w = (scaled_w - w) // 2
            img = img[start_h:start_h+h, start_w:start_w+w]
        else:
            # Need to pad
            new_img = np.zeros((h, w, 3) if len(img.shape) == 3 else (h, w), dtype=img.dtype)
            start_h = (h - scaled_h) // 2
            start_w = (w - scaled_w) // 2
            if len(img.shape) == 3:
                new_img[start_h:start_h+scaled_h, start_w:start_w+scaled_w, :] = img
            else:
                new_img[start_h:start_h+scaled_h, start_w:start_w+scaled_w] = img
            img = new_img
    
    # Apply flips
    if flip_h:
        img = cv2.flip(img, 1)  # Horizontal flip
    if flip_v:
        img = cv2.flip(img, 0)  # Vertical flip
    
    # Apply brightness adjustment
    if brightness != 0:
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness * 255, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            img = np.clip(img + brightness * 255, 0, 255).astype(np.uint8)
    
    # Convert back to original dtype
    if original_dtype == np.float32 or original_dtype == np.float64:
        img = img.astype(np.float32) / 255.0
    
    return img

def save_image(image: np.ndarray, path: str, create_dirs: bool = True) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image to save
        path: Output path
        create_dirs: Whether to create parent directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if create_dirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        # Convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            if np.max(image) <= 1.0:
                save_img = (image * 255).astype(np.uint8)
            else:
                save_img = image.astype(np.uint8)
        else:
            save_img = image
            
        # Convert to BGR for OpenCV
        if len(save_img.shape) == 3 and save_img.shape[2] == 3:
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            
        cv2.imwrite(path, save_img)
        return True
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error saving image to {path}: {e}")
        return False

def load_image_file(path: str, color_mode: str = 'rgb') -> Optional[np.ndarray]:
    """
    Load image from file.
    
    Args:
        path: Path to image file
        color_mode: Color mode ('rgb', 'bgr', 'grayscale')
        
    Returns:
        Loaded image or None if failed
    """
    try:
        if not os.path.exists(path):
            logger = get_logger(__name__)
            logger.error(f"Image file does not exist: {path}")
            return None
            
        # Read image
        img = cv2.imread(path)
        
        if img is None:
            logger = get_logger(__name__)
            logger.error(f"Failed to read image: {path}")
            return None
            
        # Convert color mode
        if color_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_mode == 'grayscale':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # else keep BGR
            
        return img
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error loading image {path}: {e}")
        return None 