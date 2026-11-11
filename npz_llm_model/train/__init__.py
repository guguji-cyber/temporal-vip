"""
训练模块初始化文件
"""
# 使用相对导入或空文件，避免循环导入
# 模块内部可以相互导入，外部应该使用绝对导入
# from .trainer import Trainer
# from .evaluator import Evaluator
from .losses import TextSimilarityLoss

__all__ = ['TextSimilarityLoss'] 