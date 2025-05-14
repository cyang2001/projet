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

        # create initial color mask
        mask = cv2.inRange(img_hsv, lower, upper)
        
        original_mask = mask.copy()
        
        # calculate image size, for dynamic adjustment of the kernel size of morphological operations
        height, width = mask.shape[:2]
        kernel_size = max(3, min(18, int(min(height, width) * 0.05)))  # dynamic kernel size, based on image size
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
        
        dilate_size = max(3, kernel_size // 2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        mask_dilated = cv2.dilate(mask_closed, kernel_dilate, iterations=1)
        
        # analyze the difference between the original mask and the processed mask
        # if the processed mask loses most of the original information, it may need a different processing strategy
        original_pixels = np.count_nonzero(original_mask)
        processed_pixels = np.count_nonzero(mask_dilated)
        
        # if the processed mask loses too much information, use a more conservative processing strategy
        # but sometimes we do need to clean most of the noise
        if original_pixels > 0 and processed_pixels / original_pixels < 0.5:
            # use a smaller kernel and fewer operations
            small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(original_mask, cv2.MORPH_OPEN, small_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_kernel)
        else:
            mask = mask_dilated
        
        return mask

    def _extract_boxes_from_mask(self, mask: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        
        height, width = mask.shape[:2]
        image_area = height * width
        
        dynamic_min_area = max(self.min_area, int(image_area * 0.001))  # at least 1% of the image
        dynamic_max_area = min(self.max_area, int(image_area * 0.007))   # at most 0.5% of the image

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < dynamic_min_area or area > dynamic_max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            aspect_ratio = w / h if h > 0 else 0
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            
            fill_ratio = area / (w * h)
            if fill_ratio < 0.3:
                continue
            
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * r * r
            circularity = area / circle_area if circle_area > 0 else 0
            if circularity < 0.4:  
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else float('inf')
            if complexity > 3.0:  
                continue
            
            confidence = (
                0.4 * fill_ratio +                
                0.3 * (1 - abs(0.75 - aspect_ratio)) +  
                0.3 * circularity                 
            )
            
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
    """
    在多个子图里可视化 MultiColorDetector 的每一步:
    1. 原图
    2. 原始HSV掩码
    3. 形态学清洗后掩码
    4. 轮廓提取
    5. ROI检测结果（应用过滤条件后）
    6. 最终NMS结果

    确保与MultiColorDetector中的方法保持完全一致的清洗和检测逻辑。
    """
    # 设置更大的图像以显示更多细节
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 原始图像检查与转换
    if image.dtype == np.float32 and image.max() <= 1.0:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)
    
    # 1. 原图 (Original)
    axes[0].imshow(image)
    axes[0].set_title("1. Original Image / Image Originale")
    axes[0].axis('off')

    # 转 BGR→HSV，与_preprocess_image方法保持一致
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 选择要展示的线路ID - 以12为例或使用配置
    line_id = "12" if "12" in detector.color_params else list(detector.color_params.keys())[0]
    
    # 获取颜色参数 - 与_extract_line_mask保持一致
    params = detector.color_params[line_id]
    lower = np.maximum(0, np.array(params["hsv_lower"]) - detector.threshold_error_dict.get(line_id, 0))
    upper = np.minimum(255, np.array(params["hsv_upper"]) + detector.threshold_error_dict.get(line_id, 0))
    
    # 2. 原始HSV掩码（未经形态学处理）
    original_mask = cv2.inRange(img_hsv, lower, upper)
    axes[1].imshow(original_mask, cmap='gray')
    axes[1].set_title(f"2. Original Mask (Line {line_id})")
    axes[1].axis('off')
    
    # 3. 形态学清洗 - 完全按照_extract_line_mask方法的实现
    # 计算图像尺寸，用于动态调整形态学操作的核大小
    height, width = original_mask.shape[:2]
    kernel_size = max(3, min(18, int(min(height, width) * 0.05)))  # 动态核大小，基于图像尺寸
    
    # 创建形态学操作的核
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 应用开操作来移除噪点
    mask_opened = cv2.morphologyEx(original_mask, cv2.MORPH_OPEN, kernel_open)
    
    # 应用闭操作来填充孔洞
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
    
    # 轻微膨胀以连接近邻区域
    dilate_size = max(3, kernel_size // 2)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    mask_dilated = cv2.dilate(mask_closed, kernel_dilate, iterations=1)
    
    # 分析原始掩码和处理后掩码的差异
    original_pixels = np.count_nonzero(original_mask)
    processed_pixels = np.count_nonzero(mask_dilated)
    
    # 根据差异选择最终掩码，与_extract_line_mask方法相同
    if original_pixels > 0 and processed_pixels / original_pixels < 0.5:
        # 使用更小的核和更少的操作
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(original_mask, cv2.MORPH_OPEN, small_kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, small_kernel)
        process_text = "使用保守处理（信息损失>50%）"
    else:
        cleaned_mask = mask_dilated
        process_text = "使用标准处理"
    
    axes[2].imshow(cleaned_mask, cmap='gray')
    axes[2].set_title(f"3. Cleaned Mask / {process_text}")
    axes[2].axis('off')
    
    # 4. 轮廓提取 - 使用与detector.detect完全相同的逻辑
    contour_img = image.copy()
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 动态计算面积阈值，与_extract_boxes_from_mask保持一致
    image_area = height * width
    dynamic_min_area = max(detector.min_area, int(image_area * 0.001))  # 至少占图像的0.1%
    dynamic_max_area = min(detector.max_area, int(image_area * 0.3))    # 最多占图像的30%
    
    # 绘制所有轮廓并显示其属性
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 计算与_extract_boxes_from_mask相同的特征
        aspect_ratio = w / h if h > 0 else 0
        fill_ratio = area / (w * h) if w * h > 0 else 0
        
        # 计算圆度
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * r * r
        circularity = area / circle_area if circle_area > 0 else 0
        
        # 计算复杂度
        perimeter = cv2.arcLength(cnt, True)
        complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else float('inf')
        
        # 在图像上绘制轮廓矩形
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色框
        
        # 确定轮廓是否通过检测标准，用于颜色编码
        passes_area = dynamic_min_area <= area <= dynamic_max_area
        passes_aspect = detector.min_aspect_ratio <= aspect_ratio <= detector.max_aspect_ratio
        passes_fill = fill_ratio >= 0.3
        passes_circularity = circularity >= 0.4
        passes_complexity = complexity <= 3.0
        passes_all = passes_area and passes_aspect and passes_fill and passes_circularity and passes_complexity
        
        # 显示更详细的轮廓信息
        feature_text = f"A:{area:.0f} R:{aspect_ratio:.2f} F:{fill_ratio:.2f}\nC:{circularity:.2f} X:{complexity:.2f}"
        box_color = 'lime' if passes_all else 'red'
        
        # 显示每个轮廓的信息
        axes[3].text(x, y-5, feature_text, color='white', fontsize=7, 
                     bbox=dict(facecolor=box_color, alpha=0.7))
    
    axes[3].imshow(contour_img)
    axes[3].set_title(f"4. All Contours ({len(contours)})")
    axes[3].axis('off')
    
    # 5. 应用过滤后的轮廓 - 使用与_extract_boxes_from_mask完全相同的逻辑
    filtered_img = image.copy()
    boxes = []
    valid_contours = 0
    
    # 重新实现_extract_boxes_from_mask的逻辑来显示过滤后的框
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 面积筛选
        if area < dynamic_min_area or area > dynamic_max_area:
            continue
        
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 计算长宽比
        aspect_ratio = w / h if h > 0 else 0
        if not (detector.min_aspect_ratio <= aspect_ratio <= detector.max_aspect_ratio):
            continue
        
        # 计算填充率
        fill_ratio = area / (w * h)
        if fill_ratio < 0.3:
            continue
        
        # 计算圆度
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * r * r
        circularity = area / circle_area if circle_area > 0 else 0
        if circularity < 0.4:
            continue
        
        # 检查轮廓的复杂度
        perimeter = cv2.arcLength(cnt, True)
        complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else float('inf')
        if complexity > 3.0:
            continue
        
        # 使用多种特征计算置信度
        confidence = (
            0.4 * fill_ratio +                # 填充率
            0.3 * (1 - abs(0.75 - aspect_ratio)) +  # 接近理想长宽比的程度
            0.3 * circularity                 # 圆度
        )
        
        boxes.append((x, y, x + w, y + h, confidence))
        valid_contours += 1
        
        # 在图像上绘制过滤后的框
        cv2.rectangle(filtered_img, (x, y), (x + w, y + h), (255, 165, 0), 2)  # 橙色框
        axes[4].text(x, y-5, f"{confidence:.2f}", color='white', fontsize=8, 
                    bbox=dict(facecolor='orange', alpha=0.7))
    
    axes[4].imshow(filtered_img)
    axes[4].set_title(f"5. Filtered Boxes ({valid_contours}/{len(contours)})")
    axes[4].axis('off')
    
    # 6. 最终NMS结果 - 使用detector.detect来展示最终结果
    final_img = image.copy()
    detections = detector.detect(image)
    
    # 按照线路分组展示检测结果
    line_detections = {}
    for det in detections:
        line_id = det["line_id"]
        if line_id not in line_detections:
            line_detections[line_id] = []
        line_detections[line_id].append(det)
    
    # 为不同线路使用不同颜色
    cmap = plt.cm.get_cmap('tab10', 10)
    
    for i, (line_id, line_dets) in enumerate(line_detections.items()):
        color_idx = i % 10
        color = tuple(int(c*255) for c in cmap(color_idx)[:3])  # 转换为BGR
        
        for det in line_dets:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            
            # 在最终图像上绘制框
            cv2.rectangle(final_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色框
            axes[5].text(x1, y1-5, f"Line {line_id}: {confidence:.2f}", color='white', fontsize=8,
                       bbox=dict(facecolor=cmap(color_idx), alpha=0.7))
    
    axes[5].imshow(final_img)
    axes[5].set_title(f"6. Final Detections ({len(detections)})")
    axes[5].axis('off')
    
    # 紧凑布局
    plt.tight_layout()
    plt.show()
    
    # 显示处理统计信息
    print(f"检测线路 {line_id} 统计:")
    print(f"  - 原始掩码非零像素: {np.count_nonzero(original_mask)}")
    print(f"  - 清洗后掩码非零像素: {np.count_nonzero(cleaned_mask)}")
    print(f"  - 识别到的轮廓数: {len(contours)}")
    print(f"  - 筛选后的框数: {len(boxes)}")
    print(f"  - 最终检测结果数: {len(detections)}")
    
    return detections  # 返回检测结果方便进一步分析

def refine_line_mask(mask: np.ndarray) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    close_kernel = open_kernel
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.dilate(closed, dilate_kernel, iterations=1)

    return cleaned