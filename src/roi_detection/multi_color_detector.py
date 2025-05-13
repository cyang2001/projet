from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import cv2
from omegaconf import DictConfig
import logging
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

from .base import BaseDetector
from utils.utils import get_logger

class MultiColorDetector(BaseDetector):
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        super().__init__(cfg, logger)

        self.min_area = cfg.get("min_area", 300)
        self.max_area = cfg.get("max_area", 20000)
        self.min_aspect_ratio = cfg.get("min_aspect_ratio", 0.5)
        self.max_aspect_ratio = cfg.get("max_aspect_ratio", 2.0)

        self.color_params = self._load_color_params(cfg)
        self.threshold_error_dict = cfg.get("threshold_error_dict", {})
        self.debug = cfg.get("debug", False)

    def _load_color_params(self, cfg: DictConfig) -> Dict:
        params_dir = cfg.get("params_dir", "")
        params_path = os.path.join(params_dir, "color_params.json")
        if params_path and os.path.exists(params_path):
            try:
                self.logger.info(f"Loading color parameters from {params_path}")
                with open(params_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load color parameters from {params_path}: {e}")

        self.logger.info("Using default color parameters for metro lines")
        return {}  

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        img_hsv = self._preprocess_image(image)
        detections = []

        for line_id in self.color_params:
            mask = self._extract_line_mask(img_hsv, line_id)
            boxes = self._extract_boxes_from_mask(mask)
            for x1, y1, x2, y2, conf in boxes:
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "line_id": line_id,
                    "confidence": conf
                })

        return self._nms_by_line(detections)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided for preprocessing")

        if image.dtype == np.float32 and image.max() <= 1.0:
            img_rgb_uint8 = (image * 255).astype(np.uint8)
        else:
            img_rgb_uint8 = image.astype(np.uint8)

        img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return img_hsv

    def _extract_line_mask(self, img_hsv: np.ndarray, line_id: str) -> np.ndarray:
        params = self.color_params[line_id]
        lower = np.maximum(0, np.array(params["hsv_lower"]) - self.threshold_error_dict.get(line_id, 0))
        upper = np.minimum(255, np.array(params["hsv_upper"]) + self.threshold_error_dict.get(line_id, 0))

        mask = cv2.inRange(img_hsv, lower, upper)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)

        return mask

    def _extract_boxes_from_mask(self, mask: np.ndarray) -> List[Tuple[int, int, int, int, float]]:

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            fill_ratio = area / (w * h)
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            if fill_ratio < 0.3:
                continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * r * r
            circularity = area / circle_area
            if circularity < 0.5:
                continue

            confidence = fill_ratio
            boxes.append((x, y, x + w, y + h, confidence))

        return boxes

    def _nms_by_line(self, detections: List[Dict[str, Any]], iou_thresh=0.5) -> List[Dict[str, Any]]:
        grouped = defaultdict(list)
        for det in detections:
            grouped[det['line_id']].append(det)

        final = []
        for line_id, group in grouped.items():
            final.extend(self._apply_nms_for_group(group, iou_thresh))
        return final

    def _apply_nms_for_group(self, regions: List[Dict[str, Any]], iou_threshold=0.5) -> List[Dict[str, Any]]:
        if not regions:
            return []

        boxes = np.array([r["bbox"] + [r["confidence"]] for r in regions])
        scores = boxes[:, 4]
        indices = np.argsort(scores)[::-1]
        keep = []

        while indices.size > 0:
            i = indices[0]
            keep.append(i)

            overlaps = self._calculate_iou(boxes[i, :4], boxes[indices[1:], :4])
            inds = np.where(overlaps <= iou_threshold)[0]
            indices = indices[inds + 1]

        return [regions[i] for i in keep]

    def _calculate_iou(self, box, boxes):
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (area_box + area_boxes - inter + 1e-6)
        return iou

    def update_params(self, params: Dict[str, Any]) -> None:
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
    # 1. 原图 (Original)
    print("image dtype:", image.dtype)
    print("image shape:", image.shape)
    print("image[100, 100]:", image[100, 100])
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original / Originale")
    axes[0].axis('off')

    # 转 BGR→HSV
    img_rgb_uint8 = (image * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    print("img_hsv[100, 100]:", img_hsv[100, 100])
    # 2. 所有线路的 HSV 掩膜累加 (Combined HSV Mask)
    combined_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    #for params in detector.color_params.values():
    #    lower = np.array(params["hsv_lower"])
    #    upper = np.array(params["hsv_upper"])
    #    mask = cv2.inRange(img_hsv, lower, upper)
    #    combined_mask = cv2.bitwise_or(combined_mask, mask)
    mask12_lower = np.array(detector.color_params["12"]["hsv_lower"])
    mask12_upper = np.array(detector.color_params["12"]["hsv_upper"])
    print(detector.threshold_error_dict)
    threshold_error = detector.threshold_error_dict["12"]
    mask12_lower = np.maximum(0, mask12_lower - threshold_error)
    mask12_upper = np.minimum(255, mask12_upper + threshold_error)
    print(mask12_lower, mask12_upper)

    mask = cv2.inRange(img_hsv, mask12_lower, mask12_upper)
    print("Non-zero pixels in mask:", np.count_nonzero(mask))
    #combined_mask = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("HSV Mask / Masque HSV")
    axes[1].axis('off')

    # 3. 形态学清洗 (Morphological Open+Close)
    cleaned = refine_line_mask(mask)
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

def refine_line_mask(mask: np.ndarray) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    close_kernel = open_kernel
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.dilate(closed, dilate_kernel, iterations=1)

    return cleaned