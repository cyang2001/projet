"""
ROI detection package for metro line recognition.
"""

from .base import BaseDetector, get_detector
from .multi_color_detector import MultiColorDetector, optimize_color_parameters

__all__ = [
    'BaseDetector',
    'get_detector',
    'MultiColorDetector',
    'optimize_color_parameters'
] 