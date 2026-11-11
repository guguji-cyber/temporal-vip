"""
模型模块
"""
from .base_model import BaseModel
# from .feature_fusion import FeatureFusion  # 已废弃
from .contrastive_module import MultiGranularityContrastive
from .temporal_alignment import TemporalAwareAlignment
from .enhanced_fusion import EnhancedMultiModalFusion
from .enhanced_transformer_model import EnhancedTransformerModel

__all__ = [
    'BaseModel',
    # 'FeatureFusion',  # 已废弃
    'MultiGranularityContrastive',
    'TemporalAwareAlignment',
    'EnhancedMultiModalFusion',
    'EnhancedTransformerModel'
] 