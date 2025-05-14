import logging
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import DictConfig
from typing import List, Optional, Tuple, Dict, Any, Union
import cv2
from PIL import Image
import time

from src.data.dataset import MetroDataset
from src.roi_detection.base import get_detector
from src.classification.base import get_classifier
from src.preprocessing.preprocessor import PreprocessingPipeline
from utils.utils import get_logger, ensure_dir

class MetroDemoPipeline:
    """
    Paris Metro Line Recognition Demo Pipeline
    
    Processes single images or batches of images and visualizes the results.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the demo pipeline.
        
        Args:
            cfg: Configuration object
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.cfg = cfg
        self._validate_config()
        self._init_components()
        
        # 创建结果目录
        self.output_dir = self.cfg.get("output_dir", "results/demo")
        ensure_dir(self.output_dir)
        
        # 设置可视化选项
        self.view_images = self.cfg.mode.demo.get("view_images", True)
        self.save_results = self.cfg.mode.demo.get("save_results", False)
        self.show_debug_info = self.cfg.mode.demo.get("show_debug_info", False)
        
        # 创建时间戳目录，避免覆盖之前的结果
        if self.save_results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.output_dir = os.path.join(self.output_dir, timestamp)
            ensure_dir(self.output_dir)
            self.logger.info(f"Results will be saved to {self.output_dir}")
    
    def _validate_config(self):
        """
        Validate the configuration parameters.
        """
        required_configs = {
            'roi_detection': "ROI detector configuration",
            'classification': "Classification configuration",
            'preprocessing': "Preprocessing configuration"
        }
        
        for key, description in required_configs.items():
            if not hasattr(self.cfg, key):
                raise ValueError(f"Configuration missing: {key} ({description})")
                
        if not hasattr(self.cfg.mode, 'demo'):
            raise ValueError("Configuration missing: mode.demo")
            
        if not hasattr(self.cfg.mode.demo, 'input_path'):
            raise ValueError("Configuration missing: mode.demo.input_path")
            
        self.logger.info("Configuration validation passed")
    
    def _init_components(self):
        """
        Initialize pipeline components.
        """
        try:
            self.logger.info("Initializing demo components...")
            
            # Initialize preprocessor
            self.preprocessor = PreprocessingPipeline(
                cfg=self.cfg.preprocessing
            )
            
            # Initialize detector
            self.detector = get_detector(
                cfg=self.cfg.roi_detection,
                logger=self.logger
            )
            
            # Initialize classifier
            self.classifier = get_classifier(
                cfg=self.cfg.classification,
                logger=self.logger
            )
            
            self.logger.info("Demo components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run(self):
        """
        Run the demo pipeline.
        
        Process single image or all images in a directory based on configuration.
        """
        self.logger.info("=== Starting Demo Pipeline ===")
        
        input_path = self.cfg.mode.demo.input_path
        batch_mode = self.cfg.mode.demo.get("batch_mode", False)
        
        if batch_mode:
            self._process_batch(input_path)
        else:
            self._process_single(input_path)
            
        self.logger.info("=== Demo Completed ===")
    
    def _process_single(self, image_path: str):
        """
        Process a single image.
        
        Args:
            image_path: Path to the image file
        """
        self.logger.info(f"Processing single image: {image_path}")
        
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            return
        
        try:
            # 加载并预处理图像
            start_time = time.time()
            image, preprocessed = self._load_and_preprocess_image(image_path)
            preprocess_time = time.time() - start_time
            
            # 检测ROI
            start_time = time.time()
            roi_results = self.detector.detect(preprocessed)
            detection_time = time.time() - start_time
            
            self.logger.info(f"Detected {len(roi_results)} potential metro signs")
            
            # 分类ROI
            start_time = time.time()
            results = self._classify_rois(preprocessed, roi_results)
            classification_time = time.time() - start_time
            
            # 记录处理时间
            processing_times = {
                'preprocess': preprocess_time,
                'detection': detection_time,
                'classification': classification_time,
                'total': preprocess_time + detection_time + classification_time
            }
            
            # 可视化结果
            if self.view_images:
                self._visualize_results(image, results, os.path.basename(image_path), processing_times)
            
            # 保存结果
            if self.save_results:
                output_path = os.path.join(self.output_dir, f"demo_{os.path.basename(image_path)}")
                self._save_visualization(image, results, output_path, processing_times)
                self._save_detection_data(results, os.path.splitext(output_path)[0] + "_data.json")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def _load_and_preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载并预处理图像。
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Tuple of (original_image, preprocessed_image)
        """
        # 加载图像
        try:
            image = np.array(Image.open(image_path).convert('RGB'))
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            # 尝试使用OpenCV加载
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                raise ValueError(f"Failed to load image {image_path}")
            image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            
        self.logger.info(f"Loaded image with shape {image.shape}")
        
        # 转换为浮点型
        image_float = image.astype(np.float32) / 255.0
        
        # 预处理
        preprocessed = self.preprocessor.process(image_float)
        self.logger.info(f"Preprocessed image shape: {preprocessed.shape}")
        
        return image, preprocessed
    
    def _classify_rois(self, preprocessed: np.ndarray, roi_results: List[Dict]) -> List[Dict]:
        """
        对检测到的ROI进行分类。
        
        Args:
            preprocessed: 预处理后的图像
            roi_results: ROI检测结果
            
        Returns:
            分类结果列表
        """
        results = []
        for roi in roi_results:
            try:
                # 提取ROI坐标
                bbox = roi["bbox"]
                x1, y1, x2, y2 = bbox
                
                # 确保坐标在图像范围内
                h, w = preprocessed.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    self.logger.warning(f"Invalid ROI coordinates: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                # 提取ROI区域
                roi_img = preprocessed[y1:y2, x1:x2]
                
                if roi_img.size == 0:
                    self.logger.warning(f"Empty ROI: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                # 分类
                class_id, confidence = self.classifier.predict(roi_img)
                
                if class_id != -1:  # 有效分类
                    result = {
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id,
                        'confidence': confidence,
                        'roi_confidence': roi.get('confidence', 0.0),
                        'line_id': roi.get('line_id', '')
                    }
                    results.append(result)
                    self.logger.info(f"Detected metro line {class_id} with confidence {confidence:.4f}")
            except Exception as e:
                self.logger.error(f"Error classifying ROI: {e}")
                
        return results
    
    def _process_batch(self, directory: str):
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
        """
        self.logger.info(f"Processing images in directory: {directory}")
        
        if not os.path.isdir(directory):
            self.logger.error(f"Directory not found: {directory}")
            return
        
        # Get all image files
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
            image_files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
        
        self.logger.info(f"Found {len(image_files)} images")
        
        if not image_files:
            self.logger.warning(f"No images found in {directory}")
            return
        
        # Process each image
        all_results = []
        for image_path in image_files:
            try:
                self.logger.info(f"Processing {os.path.basename(image_path)}")
                
                # 使用与_process_single相同的处理流程
                image_results = self._process_single(image_path)
                
                # 添加图像路径信息
                for result in image_results:
                    result['image_path'] = image_path
                    all_results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}")
        
        # Generate summary report
        self._generate_summary(all_results)
    
    def _visualize_results(self, image: np.ndarray, results: List[Dict], title: str = "", processing_times: Optional[Dict] = None):
        """
        Visualize detection results.
        
        Args:
            image: Original image
            results: List of detection results
            title: Optional title for visualization
            processing_times: Optional processing time information
        """
        if not self.view_images:
            return
            
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # Color map for different classes
        cmap = plt.cm.get_cmap('tab10', 14)  # 增加颜色映射范围
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            class_id = result['class_id']
            confidence = result['confidence']
            roi_confidence = result.get('roi_confidence', 0.0)
            
            # Choose color based on class_id
            color_idx = (class_id - 1) % 14  # 确保颜色映射正确
            color = cmap(color_idx)
            
            # Draw rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                            edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # 添加标签，包括更多信息
            label_text = f"Line {class_id} ({confidence:.2f})"
            if self.show_debug_info:
                label_text += f"\nROI conf: {roi_confidence:.2f}"
                
            plt.text(x1, y1-10, label_text,
                    color='white', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.7))
        
        # 添加处理时间信息
        if processing_times is not None and self.show_debug_info:
            info_text = (
                f"Preprocess: {processing_times['preprocess']:.3f}s\n"
                f"Detection: {processing_times['detection']:.3f}s\n"
                f"Classification: {processing_times['classification']:.3f}s\n"
                f"Total: {processing_times['total']:.3f}s"
            )
            plt.figtext(0.02, 0.02, info_text, color='black', 
                       backgroundcolor='white', fontsize=9)
        
        plt.title(f"Detection Results - {title}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _save_visualization(self, image: np.ndarray, results: List[Dict], output_path: str, processing_times: Optional[Dict] = None):
        """
        Save visualization to file.
        
        Args:
            image: Original image
            results: List of detection results
            output_path: Path to save visualization
            processing_times: Optional processing time information
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # Color map for different classes
        cmap = plt.cm.get_cmap('tab10', 14)
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            class_id = result['class_id']
            confidence = result['confidence']
            roi_confidence = result.get('roi_confidence', 0.0)
            
            # Choose color based on class_id
            color_idx = (class_id - 1) % 14
            color = cmap(color_idx)
            
            # Draw rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                            edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # 添加标签
            label_text = f"Line {class_id} ({confidence:.2f})"
            if self.show_debug_info:
                label_text += f"\nROI conf: {roi_confidence:.2f}"
                
            plt.text(x1, y1-10, label_text,
                    color='white', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.7))
        
        # 添加处理时间信息
        if processing_times is not None and self.show_debug_info:
            info_text = (
                f"Preprocess: {processing_times['preprocess']:.3f}s\n"
                f"Detection: {processing_times['detection']:.3f}s\n"
                f"Classification: {processing_times['classification']:.3f}s\n"
                f"Total: {processing_times['total']:.3f}s"
            )
            plt.figtext(0.02, 0.02, info_text, color='black', 
                       backgroundcolor='white', fontsize=9)
        
        plt.title(f"Detection Results")
        plt.axis('off')
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        ensure_dir(output_dir)
        
        # Save figure
        plt.savefig(output_path, dpi=200)
        plt.close()
        
        self.logger.info(f"Visualization saved to {output_path}")
    
    def _save_detection_data(self, results: List[Dict], output_path: str):
        """
        Save detection data to JSON file.
        
        Args:
            results: List of detection results
            output_path: Path to save data
        """
        import json
        
        # 转换为可序列化的格式
        serializable_results = []
        for r in results:
            serializable_result = {
                'bbox': list(r['bbox']),
                'class_id': int(r['class_id']),
                'confidence': float(r['confidence']),
                'roi_confidence': float(r.get('roi_confidence', 0.0)),
                'line_id': r.get('line_id', '')
            }
            serializable_results.append(serializable_result)
        
        # 保存到文件
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        self.logger.info(f"Detection data saved to {output_path}")
    
    def _generate_summary(self, all_results: List[Dict]):
        """
        Generate summary of batch processing results.
        
        Args:
            all_results: List of all detection results
        """
        if not all_results:
            self.logger.warning("No results to summarize")
            return
            
        self.logger.info("=== Summary ===")
        
        # 统计基本信息
        processed_images = set([r['image_path'] for r in all_results if 'image_path' in r])
        self.logger.info(f"Processed {len(processed_images)} images")
        self.logger.info(f"Detected {len(all_results)} metro signs")
        
        # 按类别统计
        class_counts = {}
        for result in all_results:
            class_id = result['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        self.logger.info("Detections by class:")
        for class_id in range(1, 15):  # 1-14线路
            count = class_counts.get(class_id, 0)
            self.logger.info(f"  Line {class_id}: {count}")
        
        # 计算每个类别的平均置信度
        class_confidences = {}
        for result in all_results:
            class_id = result['class_id']
            if class_id not in class_confidences:
                class_confidences[class_id] = []
            class_confidences[class_id].append(result['confidence'])
        
        self.logger.info("Average confidence by class:")
        for class_id, confidences in class_confidences.items():
            avg_confidence = sum(confidences) / len(confidences)
            self.logger.info(f"  Line {class_id}: {avg_confidence:.4f}")
        
        # 创建汇总可视化
        if self.save_results:
            # 类别分布图
            plt.figure(figsize=(12, 6))
            
            # 准备数据，确保包括所有线路
            line_ids = list(range(1, 15))  # 1-14线路
            counts = [class_counts.get(line_id, 0) for line_id in line_ids]
            
            # 创建柱状图
            bars = plt.bar(
                [f"Line {line_id}" for line_id in line_ids],
                counts,
                color=[plt.cm.get_cmap('tab10', 14)((i-1) % 14) for i in line_ids]
            )
            
            # 在柱状图上添加数值标签
            for bar, count in zip(bars, counts):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    str(count),
                    ha='center',
                    va='bottom'
                )
            
            plt.title("Metro Line Detections Distribution")
            plt.xlabel("Line")
            plt.ylabel("Count")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 保存图表
            summary_path = os.path.join(self.output_dir, "class_distribution.png")
            plt.savefig(summary_path, dpi=200)
            plt.close()
            
            self.logger.info(f"Summary chart saved to {summary_path}")
            
            # 置信度分布图
            plt.figure(figsize=(12, 6))
            
            # 准备数据
            valid_line_ids = sorted(class_confidences.keys())
            avg_confidences = [np.mean(class_confidences[i]) for i in valid_line_ids]
            min_confidences = [min(class_confidences[i]) for i in valid_line_ids]
            max_confidences = [max(class_confidences[i]) for i in valid_line_ids]
            
            # 绘制平均置信度
            x = np.arange(len(valid_line_ids))
            plt.bar(
                x,
                avg_confidences,
                color=[plt.cm.get_cmap('tab10', 14)((i-1) % 14) for i in valid_line_ids],
                alpha=0.7
            )
            
            # 添加最小/最大置信度误差线
            plt.errorbar(
                x,
                avg_confidences,
                yerr=[
                    [a - b for a, b in zip(avg_confidences, min_confidences)],
                    [b - a for a, b in zip(avg_confidences, max_confidences)]
                ],
                fmt='none',
                capsize=5,
                color='black'
            )
            
            plt.title("Classification Confidence by Line")
            plt.xlabel("Line")
            plt.ylabel("Confidence")
            plt.xticks(x, [f"Line {i}" for i in valid_line_ids])
            plt.ylim(0, 1.1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 保存图表
            confidence_path = os.path.join(self.output_dir, "confidence_distribution.png")
            plt.savefig(confidence_path, dpi=200)
            plt.close()
            
            self.logger.info(f"Confidence chart saved to {confidence_path}")
            
            # 保存汇总数据
            summary_data = {
                "processed_images": len(processed_images),
                "total_detections": len(all_results),
                "class_counts": {str(k): v for k, v in class_counts.items()},
                "avg_confidences": {str(k): float(np.mean(v)) for k, v in class_confidences.items()},
                "min_confidences": {str(k): float(min(v)) for k, v in class_confidences.items()},
                "max_confidences": {str(k): float(max(v)) for k, v in class_confidences.items()}
            }
            
            import json
            summary_json_path = os.path.join(self.output_dir, "summary.json")
            with open(summary_json_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
                
            self.logger.info(f"Summary data saved to {summary_json_path}")


def main(cfg: DictConfig):
    """
    Main entry point for demo pipeline.
    
    Args:
        cfg: Configuration object
    """
    logger = get_logger(__name__)
    
    try:
        # Create demo pipeline
        pipeline = MetroDemoPipeline(
            cfg=cfg,
            logger=logger
        )
        
        # Run demo
        pipeline.run()
        
        logger.info("Demo pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Demo pipeline execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc()) 