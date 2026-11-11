"""
特征提取模块
"""
from .static_extractor import StaticFeatureExtractor
from .dynamic_extractor import DynamicFeatureExtractor
from .text_extractor import TextFeatureExtractor
# from .facial_analyzer import FacialAnalyzer  # 已废弃

__all__ = [
    'StaticFeatureExtractor',
    'DynamicFeatureExtractor',
    'TextFeatureExtractor',
    # 'FacialAnalyzer',  # 已废弃
] 