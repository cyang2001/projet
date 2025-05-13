from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import logging
from typing import Optional, Tuple, List, Iterator, Dict, Any, Union
import scipy.io as sio
import os

from omegaconf import DictConfig
from utils.utils import get_logger
from sklearn.model_selection import train_test_split

def _clip_bbox(bbox: Tuple[int, int, int, int], max_width: int, max_height: int) -> Tuple[int, int, int, int]:
    """
    Clip bounding box coordinates to image dimensions.
    
    Args:
        bbox: Original bounding box (x1, y1, x2, y2)
        max_width: Image width
        max_height: Image height
        
    Returns:
        Clipped bounding box
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, max_width - 1))
    y1 = max(0, min(y1, max_height - 1))
    x2 = max(0, min(x2, max_width))
    y2 = max(0, min(y2, max_height))
    return x1, y1, x2, y2


class MetroDataset:
    """
    Dataset for Paris Metro pictogram recognition.
    
    Supports loading data from:
    1. Excel files (.xls/.xlsx) - Original format
    2. MATLAB files (.mat) - New format
    
    Handles data splitting according to project requirements.
    """

    def __init__(
        self,
        cfg: DictConfig,
        mode: str = 'train',
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            cfg: Configuration dictionary
            mode: 'train', 'val', or 'test' 
            logger: Optional logger
        """
        self.logger = logger or get_logger(__name__)
        self.mode = mode
        self.data_root = Path(cfg.get("data_root", "dataset"))
        # Support either Excel or MAT files
        self.data_format = cfg.get("data_format", "excel").lower()  # 'excel' or 'mat'
        
        # Excel file paths
        self.train_xls = self.data_root / cfg.get("train_xls", "Apprentissage.xls")
        self.test_xls = self.data_root / cfg.get("test_xls", "Test.xls")
        
        # MAT file paths
        self.train_mat = self.data_root / cfg.get("train_mat", "Apprentissage.mat")
        self.test_mat = self.data_root / cfg.get("test_mat", "Test.mat")
        
        # Image format
        self.img_format = cfg.get("img_format", "JPG")
        
        self.val_split = cfg.get("val_split")
        self.random_seed = cfg.get("random_seed")
        # Load annotations based on format
        if self.data_format == 'excel':
            self._load_excel_data()
        else:  # mat format
            self._load_mat_data()
            
        # Common post-processing
        if self.df.empty:
            self.logger.error(f"No annotation data for mode '{self.mode}'")
            raise ValueError(f"No annotation data for mode '{self.mode}'")

        self.df['image_path'] = self.df['image_id'].apply(self._resolve_image_path)

        self.class_counts = self.df['class_id'].value_counts().to_dict()
        self.logger.info(
            f"Mode={self.mode}: loaded {len(self.df)} samples from {self.annotations.shape[0]} annotations"
        )
    
    def _load_excel_data(self) -> None:
        """
        Load data from Excel files.
        """
        if self.mode in {'train', 'val'}:
            self.annotations = self._load_excel(self.train_xls)
            self.df = self._split_train_val(self.annotations)
        elif self.mode == 'test':
            self.annotations = self._load_excel(self.test_xls)
            self.df = self.annotations.copy()
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'train', 'val' or 'test'.")
    
    def _load_mat_data(self) -> None:
        """
        Load data from MAT files.
        """
        if self.mode in {'train', 'val'}:
            self.annotations = self._load_mat_to_df(self.train_mat)
            self.df = self._split_train_val(self.annotations, self.mode)
        elif self.mode == 'test':
            self.annotations = self._load_mat_to_df(self.test_mat)
            self.df = self.annotations.copy()
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'train', 'val' or 'test'.")

    def _load_excel(self, path: Path) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            path: Path to Excel file
            
        Returns:
            DataFrame with annotations
        """
        if not path.exists():
            msg = f"Annotation file not found: {path}"
            self.logger.error(msg)
            raise ValueError(msg)
        try:
            df = pd.read_excel(path)
        except Exception as e:
            msg = f"Failed to read Excel {path}: {e}"
            self.logger.error(msg)
            raise ValueError(msg)
        required = {'image_id', 'y1', 'y2', 'x1', 'x2', 'class_id'}
        if not required.issubset(df.columns):
            msg = f"Missing columns in {path}: {required - set(df.columns)}"
            self.logger.error(msg)
            raise ValueError(msg)
        return df
        
    def _load_mat_to_df(self, path: Path) -> pd.DataFrame:
        """
        Load data from MAT file and convert to DataFrame.
        Author: Chen YANG, Amélioré par ChatGPT
        Args:
            path: Path to MAT file
        Returns:
            DataFrame with annotations
        """
        if not path.exists():
            msg = f"Annotation file not found: {path}"
            self.logger.error(msg)
            raise ValueError(msg)
            
        try:
            # Load .mat file
            self.logger.info(f"Loading MAT file: {path}")
            mat_data = sio.loadmat(str(path))
            
            # Check for data field (usually named 'BD' in metro line dataset)
            data_field = None
            for key in mat_data.keys():
                if key not in ['__header__', '__version__', '__globals__']:
                    if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) == 2:
                        data_field = key
                        break
            
            if data_field is None:
                # Try known field names
                for field in ['BD', 'annot', 'data']:
                    if field in mat_data:
                        data_field = field
                        break
            
            if data_field is None:
                msg = f"Could not find data array in MAT file: {path}"
                self.logger.error(msg)
                raise ValueError(msg)
            
            # Get the annotation data array
            annot_data = mat_data[data_field]
            self.logger.info(f"Found annotation data with shape: {annot_data.shape}")
            
            # Check if we have enough columns
            if annot_data.shape[1] < 6:
                msg = f"Annotation data should have at least 6 columns, but has {annot_data.shape[1]}"
                self.logger.error(msg)
                raise ValueError(msg)
            
            # Format is (image_id, y1, y2, x1, x2, class_id)
            # Create DataFrame directly from numpy array
            df = pd.DataFrame({
                'image_id': annot_data[:, 0].astype(int),
                'y1': annot_data[:, 1].astype(int),
                'y2': annot_data[:, 2].astype(int),
                'x1': annot_data[:, 3].astype(int),
                'x2': annot_data[:, 4].astype(int),
                'class_id': annot_data[:, 5].astype(int)
            })
            
            self.logger.info(f"Successfully loaded {len(df)} annotations from MAT file")
            return df
            
        except Exception as e:
            msg = f"Failed to read MAT file {path}: {e}"
            self.logger.error(msg)
            raise ValueError(msg)

    def _extract_str_from_mat(self, mat_item) -> str:
        """
        Extract string from MATLAB cell array element.
        
        Args:
            mat_item: MATLAB cell array item
            
        Returns:
            Extracted string
        """
        # Handle different ways strings can be stored in MATLAB files
        if isinstance(mat_item, np.ndarray) and mat_item.dtype.type is np.str_:
            return str(mat_item[0])
        elif isinstance(mat_item, np.ndarray) and mat_item.dtype.type is np.bytes_:
            return mat_item[0].decode('utf-8')
        elif isinstance(mat_item, np.ndarray) and mat_item.size > 0:
            if isinstance(mat_item[0], np.ndarray):
                return self._extract_str_from_mat(mat_item[0])
            else:
                return str(mat_item[0])
        else:
            return str(mat_item)
            
    def _split_train_val(
        self, df: pd.DataFrame,
        mode: str = 'train'
    ) -> pd.DataFrame:
        # 避免同一个image_id被分到两个不同的set中 造成图像泄漏
        unique_ids = df['image_id'].unique()
        train_ids, val_ids = train_test_split(unique_ids, test_size=self.val_split, random_state=self.random_seed)
        if mode == 'train':
            return df[df['image_id'].isin(train_ids)].reset_index(drop=True)
        elif mode == 'val':
            return df[df['image_id'].isin(val_ids)].reset_index(drop=True)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _resolve_image_path(self, image_id: Any) -> str:
        """
        Resolve path to image file.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Path to image file
        """
        # Original style: IM (123).JPG
        img_path = self.data_root / f"IM ({int(image_id)}).{self.img_format}"
        if img_path.exists():
            return str(img_path)
            
        # Try alternative format: image_123.jpg
        alt_path = self.data_root / f"image_{int(image_id)}.{self.img_format.lower()}"
        if alt_path.exists():
            return str(alt_path)
            
        # Try in images, train, test folders
        for folder in ["images", "train", "test"]:
            folder_path = self.data_root / folder / f"IM ({int(image_id)}).{self.img_format}"
            if folder_path.exists():
                return str(folder_path)
                
            alt_folder_path = self.data_root / folder / f"image_{int(image_id)}.{self.img_format.lower()}"
            if alt_folder_path.exists():
                return str(alt_folder_path)

        msg = f"Image for ID {image_id} not found"
        self.logger.error(msg)
        raise ValueError(msg)

    def __len__(self) -> int:
        """
        Get number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Tuple[int,int,int,int]]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (ROI image array, class ID, bounding box)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        h, w, _ = img_arr.shape

        bbox = (int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2']))
        bbox = _clip_bbox(bbox, w, h)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            msg = f"Invalid bbox {bbox} for image ID {row['image_id']}"
            self.logger.error(msg)
            raise ValueError(msg)

        roi = img_arr[y1:y2, x1:x2]
        return roi, int(row['class_id']), bbox

    def get_original(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get original image.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (original image array, image ID)
        """
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        return np.asarray(img, dtype=np.float32) / 255.0, int(row['image_id'])

    def get_image_with_annotations(self, idx: int) -> Tuple[np.ndarray, List[Tuple[int,int,int,int,int]]]:
        """
        Get image with all its annotations.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image array, list of annotations as (x1, y1, x2, y2, class_id))
        """
        img_arr, image_id = self.get_original(idx)
        subset = self.annotations[self.annotations['image_id'] == image_id]
        boxes: List[Tuple[int,int,int,int,int]] = []
        h, w = img_arr.shape[:2]
        for _, r in subset.iterrows():
            bbox = _clip_bbox((int(r['x1']), int(r['y1']), int(r['x2']), int(r['y2'])), w, h)
            boxes.append((*bbox, int(r['class_id'])))
        return img_arr, boxes

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Iterate over dataset.
        
        Yields:
            Tuple of (ROI image array, class ID)
        """
        for idx in range(len(self)):
            roi, cls, _ = self[idx]
            yield roi, cls

    def get_batch(
        self, batch_size: int = 32, shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of samples.
        
        Args:
            batch_size: Number of samples
            shuffle: Whether to shuffle samples
            
        Returns:
            Tuple of (batch of ROI images, batch of class IDs)
        """
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        chosen = indices[:batch_size]
        rois, labels = [], []
        for i in chosen:
            roi, cls, _ = self[i]
            rois.append(roi)
            labels.append(cls)
        return np.stack(rois), np.array(labels, dtype=int)

    def get_all(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get all samples as raw ROIs without stacking.
        
        Returns:
            Tuple of (list of original ROI images, array of class IDs)
            The ROIs are returned as a list since they may have different dimensions.
        """
        if len(self) == 0:
            return [], np.array([], dtype=int)
            
        rois, labels = [], []
        for idx in range(len(self)):
            roi, cls, _ = self[idx]
            rois.append(roi)
            labels.append(cls)
            
        return rois, np.array(labels, dtype=int)

    def get_class_balance_weights(self) -> Dict[int, float]:
        """
        Get class weights for balancing.
        
        Returns:
            Dictionary of {class_id: weight}
        """
        counts = self.df['class_id'].value_counts().to_dict()
        if not counts:
            return {}
        max_c = max(counts.values())
        return {cid: max_c / cnt for cid, cnt in counts.items()}

    def get_unique_classes(self) -> List[int]:
        """
        Get list of unique class IDs.
        
        Returns:
            List of class IDs
        """
        return sorted(self.df['class_id'].unique().tolist())

    def get_unique_image_ids(self) -> List[int]:
        """
        Get list of unique image IDs.
        
        Returns:
            List of image IDs
        """
        return sorted(self.df['image_id'].unique().tolist())
