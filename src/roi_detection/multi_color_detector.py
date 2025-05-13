"""
Multi-color detector for Paris Metro line signs.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import cv2
from omegaconf import DictConfig
import logging
import os
import json
import matplotlib.pyplot as plt

from src.data.dataset import MetroDataset

from .base import BaseDetector
from utils.utils import get_logger

class MultiColorDetector(BaseDetector):
    """
    Multi-color ROI detector for Paris Metro line signs.
    
    Uses specific color ranges for each of the 14 metro lines.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the multi-color detector.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        super().__init__(cfg, logger)
        
        self.min_area = cfg.get("min_area", 300)
        self.max_area = cfg.get("max_area", 20000)
        self.min_aspect_ratio = cfg.get("min_aspect_ratio", 0.5)
        self.max_aspect_ratio = cfg.get("max_aspect_ratio", 2.0)
        
        self.color_params = self._load_color_params(cfg)
        
        self.debug = cfg.get("debug", False)
    
    def _load_color_params(self, cfg: DictConfig) -> Dict:
        """
        Load color parameters for each metro line.
        
        Args:
            cfg: Configuration dictionary
            
        Returns:
            Dictionary of color parameters
        """
        params_path = cfg.get("params_dir", "")
        
        if params_path and os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load color parameters from {params_path}: {e}")
        
        self.logger.info("Using default color parameters for metro lines")
        # Todo  hard code 写在 config 里
        return {
            "1": {"hsv_lower": [14, 108, 85], "hsv_upper": [47, 218, 213]},
            "2": {"hsv_lower": [73, 64, 121], "hsv_upper": [125, 189, 210]},
            "3": {"hsv_lower": [25, 81, 80], "hsv_upper": [60, 199, 165]},
            "4": {"hsv_lower": [122, 71, 40], "hsv_upper": [153, 144, 181]},
            "5": {"hsv_lower": [10, 163, 91], "hsv_upper": [33, 192, 172]},
            "6": {"hsv_lower": [46, 11, 56], "hsv_upper": [93, 105, 182]},
            "7": {"hsv_lower": [104, 43, 113], "hsv_upper": [180, 113, 165]},
            "8": {"hsv_lower": [105, 33, 47], "hsv_upper": [160, 81, 198]},
            "9": {"hsv_lower": [28, 148, 98], "hsv_upper": [54, 199, 117]},
            "10": {"hsv_lower": [20, 151, 37], "hsv_upper": [40, 179, 186]},
            "11": {"hsv_lower": [0, 57, 91], "hsv_upper": [61, 119, 115]},
            "12": {"hsv_lower": [60, 51, 48], "hsv_upper": [104, 105, 132]},
            "13": {"hsv_lower": [71, 37, 72], "hsv_upper": [119, 133, 180]},
            "14": {"hsv_lower": [89, 67, 25], "hsv_upper": [146, 131, 205]}
        }
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect ROIs in the image.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            List of detected ROIs as (x1, y1, x2, y2, confidence)
        """
        if image is None or image.size == 0:
            self.logger.warning("Empty image provided for detection")
            return []
        
        if image.dtype == np.float32 and image.max() <= 1.0:
            img_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        all_regions = []
        
        for line_id, params in self.color_params.items():
            hsv_lower = np.array(params["hsv_lower"])
            hsv_upper = np.array(params["hsv_upper"])
            
            mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
            
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if self.debug:
                self.logger.info(f"Line {line_id}: Found {len(contours)} potential regions")
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < self.min_area or area > self.max_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    continue
                
                if area / (w * h) < 0.3:  # 区域填充率太低
                    continue
                
                confidence = area / (w * h)
                all_regions.append((x, y, x + w, y + h, confidence))
        final_boxes = []
        for box in self._apply_nms(all_regions):
            x1, y1, x2, y2, conf = box
            final_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return final_boxes
    
    def _apply_nms(self, regions, iou_threshold=0.5):
        """
        Apply non-maximum suppression to remove overlapping boxes.
        
        Args:
            regions: List of regions (x1, y1, x2, y2, confidence)
            iou_threshold: IoU threshold for overlap
            
        Returns:
            List of filtered regions
        """
        if not regions:
            return []
        
        boxes = np.array(regions)

        scores = boxes[:, 4]
        
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while indices.size > 0:
            i = indices[0]
            keep.append(i)
            
            overlaps = self._calculate_iou(boxes[i, :4], boxes[indices[1:], :4])
            
            inds = np.where(overlaps <= iou_threshold)[0]
            
            indices = indices[inds + 1]
        
        return boxes[keep].tolist()
    
    def _calculate_iou(self, box, boxes):
        """
        Calculate IoU between one box and multiple boxes.
        
        Args:
            box: Single box (x1, y1, x2, y2)
            boxes: Array of boxes (N, 4)
            
        Returns:
            Array of IoU values
        """
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        
        iou = inter / (area_box + area_boxes - inter)
        return iou
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        Update detector parameters.
        
        Args:
            params: Parameter dictionary
        """
        if 'min_area' in params:
            self.min_area = params['min_area']
        if 'max_area' in params:
            self.max_area = params['max_area']
        if 'min_aspect_ratio' in params:
            self.min_aspect_ratio = params['min_aspect_ratio']
        if 'max_aspect_ratio' in params:
            self.max_aspect_ratio = params['max_aspect_ratio']
        
        if 'color_params' in params:
            self.color_params.update(params['color_params'])
        
        self.logger.info("MultiColorDetector parameters updated")


def optimize_color_parameters(dataset, logger=None, visualize=False)->Dict[str, Any]:
    """
    Optimize color parameters based on training data.
    
    Args:
        dataset: Dataset with ground truth annotations, must implement:
                - __len__() method
                - get_image_with_annotations(idx) method that returns (image, annotations)
        logger: Optional logger
        visualize: Whether to visualize the dominant color extraction process
        
    Returns:
        Dictionary of optimized color parameters
    """
    import cv2
    import numpy as np
    
    logger = logger or get_logger(__name__)
    logger.info("Starting color parameter optimization...")
    
    optimized_params = {}
    
    color_samples = {}
    
    rois = []
    for idx in range(len(dataset)):
        image, annotations = dataset.get_image_with_annotations(idx)
        if image.dtype == np.float32 and image.max() <= 1.0:
            image_cv = (image * 255).astype(np.uint8)
        else:
            image_cv = image.astype(np.uint8)
        
        if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
            bgr = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        else:
            continue
        for annotation in annotations:
            x1, y1, x2, y2 = annotation[:4]
            roi_hsv = hsv[y1:y2, x1:x2]
            roi_bgr = bgr[y1:y2, x1:x2]
            line_id = annotation[4]
            try:
                dominant_hsv = extract_dominant_hsv(roi_hsv, K=3)
                if line_id not in color_samples:
                    color_samples[line_id] = []
                color_samples[line_id].append(dominant_hsv)
                if visualize:
                    visualize_dominant_color(image, roi_bgr, dominant_hsv, line_id, x1, y1, x2, y2)
            except Exception as e:
                logger.error(f"Error processing ROI ")
    
    for line_id, samples in color_samples.items():
        if not samples:
            continue
        
        samples_array = np.array(samples)
        
        h_values = samples_array[:, 0]
        if np.max(h_values) - np.min(h_values) > 90:
            h_values_adjusted = h_values.copy()
            if np.median(h_values) < 90:
                h_values_adjusted[h_values > 90] -= 180
            else:
                h_values_adjusted[h_values < 90] += 180
            
            avg_h = np.mean(h_values_adjusted)
            if avg_h < 0:
                avg_h += 180
            elif avg_h > 180:
                avg_h -= 180
        else:
            avg_h = np.mean(h_values)
        
        avg_s = np.mean(samples_array[:, 1])
        avg_v = np.mean(samples_array[:, 2])
        
        std_h = np.std(h_values)
        std_s = np.std(samples_array[:, 1])
        std_v = np.std(samples_array[:, 2])
        
        optimized_params[str(line_id)] = {
            "hsv_mean": (int(avg_h), int(avg_s), int(avg_v)),
            "hsv_std": (int(std_h), int(std_s), int(std_v)),
            "hsv_lower": [max(0, int(avg_h - 2 * std_h)), 
                          max(0, int(avg_s - 2 * std_s)), 
                          max(0, int(avg_v - 2 * std_v))],
            "hsv_upper": [min(180, int(avg_h + 2 * std_h)), 
                          min(255, int(avg_s + 2 * std_s)), 
                          min(255, int(avg_v + 2 * std_v))]
        }
    return optimized_params 
def visualize_dominant_color(img, roi, hsv_color, line_id, x1, y1, x2, y2):

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    edgecolor='g', facecolor='none', linewidth=2)) 
    plt.title(f"Original Image with Line {line_id} ROI")

    plt.subplot(1, 3, 2)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    plt.imshow(roi_rgb)
    plt.title(f"ROI of Line {line_id}")
    
    plt.subplot(1, 3, 3)
    color_patch = np.ones((100, 100, 3), dtype=np.uint8)
    h, s, v = hsv_color
    color_patch_hsv = np.full((100, 100, 3), (h, s, v), dtype=np.uint8)
    color_patch = cv2.cvtColor(color_patch_hsv, cv2.COLOR_HSV2BGR)
    color_patch_rgb = cv2.cvtColor(color_patch, cv2.COLOR_BGR2RGB)
    plt.imshow(color_patch_rgb)
    plt.title(f"Dominant Color\nHSV: ({h}, {s}, {v})")
    
    plt.tight_layout()
    plt.show()
def extract_dominant_hsv(
    roi_hsv: np.ndarray,
    K: int = 3,
    attempts: int = 10
) -> Tuple[int, int, int]:
    """
    从单个 ROI 的 HSV 像素中提取主峰色 (dominant HSV color / couleur HSV dominante)。

    Args:
        roi_hsv: ROI 在 HSV 空间的像素 (h, w, 3)
        K: 聚类簇数量 (clusters)
        attempts: kmeans 重启次数，选取最佳质心

    Returns:
        hsv_dominant: 三元组 (H, S, V)，代表主峰色
    """
    # 1. 准备样本 (Prepare samples / Préparer échantillons)
    pixels = roi_hsv.reshape(-1, 3).astype(np.float32)  # N × 3

    # 2. 定义 KMeans 终止条件 (KMeans criteria)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,    # 最多迭代次数 (max iterations)
        0.2     # 误差阈值 (epsilon)
    )

    # 3. 执行 K-Means 聚类
    ret, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        attempts,
        flags=cv2.KMEANS_PP_CENTERS
    ) #type: ignore
    # centers: shape (K, 3), dtype float32

    # 4. 统计每个簇的样本数 (Count cluster sizes / Taille des clusters)
    _, counts = np.unique(labels, return_counts=True)

    # 5. 选择主峰簇 (Select dominant cluster / Sélection du cluster dominant)
    dominant_idx = np.argmax(counts)
    dominant_center = centers[dominant_idx]  # HSV 坐标

    # 6. 返回整数化的 HSV 三元组
    return tuple(map(int, dominant_center))

def visualize_detection_steps(detector, image: np.ndarray):
    """
    在五个子图里可视化 MultiColorDetector 的每一步：
    1. 原图
    2. 各线路 HSV 掩膜 (sum)
    3. 形态学清洗后掩膜
    4. 轮廓高亮
    5. NMS 后的最终 ROI
    """
    # 1. 原图 (Original)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original / Originale")
    axes[0].axis('off')

    # 转 BGR→HSV
    img_bgr = cv2.cvtColor((image).astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 2. 所有线路的 HSV 掩膜累加 (Combined HSV Mask)
    #combined_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    #for params in detector.color_params.values():
    #    lower = np.array(params["hsv_lower"])
    #    upper = np.array(params["hsv_upper"])
    #    mask = cv2.inRange(img_hsv, lower, upper)
    #    combined_mask = cv2.bitwise_or(combined_mask, mask)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([60, 51, 48], np.uint8)
    upper = np.array([104,105,132], np.uint8)
    mask12 = cv2.inRange(hsv, lower, upper)
    axes[1].imshow(mask12, cmap='gray')
    axes[1].set_title("HSV Mask / Masque HSV")
    axes[1].axis('off')

    # 3. 形态学清洗 (Morphological Open+Close)
    kernel = np.ones((5,5), np.uint8)
    opened = cv2.morphologyEx( mask12, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    axes[2].imshow(cleaned, cmap='gray')
    axes[2].set_title("Cleaned Mask / Masque nettoyé")
    axes[2].axis('off')

    # 4. 轮廓 (Contours Highlight)
    cont_img = image.copy()
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(cont_img, (x,y), (x+w,y+h), (0,255,0), 2)  # green box
    axes[3].imshow(cont_img)
    axes[3].set_title("Contours / Contours")
    axes[3].axis('off')

    # 5. 非极大值抑制后 ROI (Final ROIs after NMS)
    regions = []
    # （下面简化：直接从 detector.detect 拿 final_boxes）
    final_boxes = detector.detect(image)
    final_img = image.copy()
    for x1,y1,x2,y2,conf in final_boxes:
        cv2.rectangle(final_img, (x1,y1), (x2,y2), (255,0,0), 2)  # blue box
        axes[4].text(x1, y1-5, f"{conf:.2f}", color='blue', fontsize=8)
    axes[4].imshow(final_img)
    axes[4].set_title("Final ROIs / ROI finales")
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()