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
        params_path = cfg.get("color_params_path", "")
        
        if params_path and os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load color parameters from {params_path}: {e}")
        
        self.logger.info("Using default color parameters for metro lines")
        # Todo  hard code 写在 config 里
        return {
            "1": {"hsv_lower": [20, 100, 100], "hsv_upper": [30, 255, 255]},
            "2": {"hsv_lower": [100, 100, 100], "hsv_upper": [130, 255, 255]},
            "3": {"hsv_lower": [10, 100, 50], "hsv_upper": [20, 255, 200]},
            "4": {"hsv_lower": [140, 50, 50], "hsv_upper": [160, 255, 255]},
            "5": {"hsv_lower": [10, 150, 150], "hsv_upper": [25, 255, 255]},
            "6": {"hsv_lower": [70, 50, 50], "hsv_upper": [100, 255, 255]},
            "7": {"hsv_lower": [150, 50, 100], "hsv_upper": [170, 150, 255]},
            "8": {"hsv_lower": [135, 50, 50], "hsv_upper": [150, 255, 200]},
            "9": {"hsv_lower": [30, 100, 100], "hsv_upper": [50, 255, 255]},
            "10": {"hsv_lower": [20, 100, 50], "hsv_upper": [30, 200, 200]},
            "11": {"hsv_lower": [0, 100, 50], "hsv_upper": [10, 255, 200]},
            "12": {"hsv_lower": [60, 100, 50], "hsv_upper": [80, 255, 200]},
            "13": {"hsv_lower": [90, 50, 150], "hsv_upper": [110, 150, 255]},
            "14": {"hsv_lower": [100, 150, 50], "hsv_upper": [130, 255, 150]}
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
        
        # 将图像转换为HSV
        if image.dtype == np.float32 and image.max() <= 1.0:
            # 归一化图像转为uint8
            img_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 存储所有线路的检测结果
        all_regions = []
        
        # 对每条线路进行颜色检测
        for line_id, params in self.color_params.items():
            hsv_lower = np.array(params["hsv_lower"])
            hsv_upper = np.array(params["hsv_upper"])
            
            # 创建颜色掩码
            mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
            
            # 形态学操作 - 去噪和填充孔洞
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 找到连通区域
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 调试模式 - 显示每条线路的掩码
            if self.debug:
                self.logger.info(f"Line {line_id}: Found {len(contours)} potential regions")
            
            # 处理每个连通区域
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 面积筛选
                if area < self.min_area or area > self.max_area:
                    continue
                
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 宽高比筛选
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    continue
                
                # 形状特征验证
                if area / (w * h) < 0.3:  # 区域填充率太低
                    continue
                
                # 添加到候选区域
                # 使用区域填充率作为置信度
                confidence = area / (w * h)
                all_regions.append((x, y, x + w, y + h, confidence))
        
        # 应用非极大值抑制去除重叠框
        return self._apply_nms(all_regions)
    
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
        
        # 转换为numpy数组方便处理
        boxes = np.array(regions)
        
        # 获取置信度
        scores = boxes[:, 4]
        
        # 按置信度排序
        indices = np.argsort(scores)[::-1]
        
        # 应用NMS
        keep = []
        while indices.size > 0:
            # 保留置信度最高的框
            i = indices[0]
            keep.append(i)
            
            # 计算其他框与当前框的IoU
            overlaps = self._calculate_iou(boxes[i, :4], boxes[indices[1:], :4])
            
            # 找到不与当前框重叠的框
            inds = np.where(overlaps <= iou_threshold)[0]
            
            # 更新索引
            indices = indices[inds + 1]
        
        # 返回保留的框
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
        # 框的面积
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 计算交集
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])
        
        # 交集的宽高
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # 交集面积
        inter = w * h
        
        # 计算IoU
        iou = inter / (area_box + area_boxes - inter)
        return iou
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        Update detector parameters.
        
        Args:
            params: Parameter dictionary
        """
        # 更新通用参数
        if 'min_area' in params:
            self.min_area = params['min_area']
        if 'max_area' in params:
            self.max_area = params['max_area']
        if 'min_aspect_ratio' in params:
            self.min_aspect_ratio = params['min_aspect_ratio']
        if 'max_aspect_ratio' in params:
            self.max_aspect_ratio = params['max_aspect_ratio']
        
        # 更新线路颜色参数
        if 'color_params' in params:
            self.color_params.update(params['color_params'])
        
        self.logger.info("MultiColorDetector parameters updated")


def optimize_color_parameters(dataset, logger=None):
    """
    Optimize color parameters based on training data.
    
    Args:
        dataset: Dataset with ground truth annotations
        logger: Optional logger
        
    Returns:
        Dictionary of optimized color parameters
    """
    import cv2
    import numpy as np
    
    logger = logger or get_logger(__name__)
    logger.info("Starting color parameter optimization...")
    
    # 初始化颜色参数字典
    optimized_params = {}
    
    # 按线路ID分组收集颜色样本
    color_samples = {}
    
    # 遍历数据集
    for idx in range(len(dataset)):
        image, annotations = dataset.get_image_with_annotations(idx)
        
        # 转换为HSV
        if image.dtype == np.float32 and image.max() <= 1.0:
            image_cv = (image * 255).astype(np.uint8)
        else:
            image_cv = image.astype(np.uint8)
        
        if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
            bgr = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        else:
            # 灰度图无法提取颜色信息
            continue
        
        # 分析每个标注
        for ann in annotations:
            x1, y1, x2, y2, line_id = ann
            
            # 确保边界框有效
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                continue
            
            # 提取ROI
            roi_hsv = hsv[y1:y2, x1:x2]
            
            if roi_hsv.size == 0:
                continue
            
            # 计算HSV均值
            h_mean, s_mean, v_mean = cv2.mean(roi_hsv)[:3]
            
            # 将颜色样本按线路ID分组
            line_id_str = str(int(line_id))
            if line_id_str not in color_samples:
                color_samples[line_id_str] = []
            
            color_samples[line_id_str].append((h_mean, s_mean, v_mean))
    
    # 对每条线路优化颜色参数
    for line_id, samples in color_samples.items():
        if not samples:
            logger.warning(f"Line {line_id}: No color samples available")
            continue
        
        hsv_array = np.array(samples)
        
        # 计算HSV范围（均值±2倍标准差）
        hsv_mean = np.mean(hsv_array, axis=0)
        hsv_std = np.std(hsv_array, axis=0)
        
        hsv_lower = np.clip(hsv_mean - 2 * hsv_std, 0, 255).astype(int)
        hsv_upper = np.clip(hsv_mean + 2 * hsv_std, 0, 255).astype(int)
        
        # 处理H通道的环状特性
        if hsv_upper[0] - hsv_lower[0] > 90:
            h_values = hsv_array[:, 0]
            
            # 调整H值，处理跨越0/180边界的情况
            if np.median(h_values) < 90:
                h_values[h_values > 90] -= 180
            else:
                h_values[h_values < 90] += 180
            
            h_mean = np.mean(h_values)
            h_std = np.std(h_values)
            
            if h_mean < 0:
                h_mean += 180
            elif h_mean > 180:
                h_mean -= 180
                
            hsv_lower[0] = max(0, int(h_mean - 2 * h_std))
            hsv_upper[0] = min(180, int(h_mean + 2 * h_std))
        
        # 保存该线路的颜色参数
        optimized_params[line_id] = {
            "hsv_lower": hsv_lower.tolist(),
            "hsv_upper": hsv_upper.tolist()
        }
        
        logger.info(f"Line {line_id}: Optimized HSV range: {hsv_lower} to {hsv_upper}")
    
    return optimized_params 