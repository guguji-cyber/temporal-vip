"""
评估指标计算模块
"""
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class Metrics:
    """评估指标计算类"""
    
    def __init__(self):
        """初始化评估指标计算器"""
        self.reset()
        
    def reset(self):
        """重置所有指标"""
        self.total_correct = 0
        self.total_samples = 0
        self.predictions = []
        self.targets = []
        self.rank1_accuracy = 0.0
        
    def update(self, preds, targets):
        """
        更新指标
        
        参数:
            preds: 预测值
            targets: 目标值
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()
            
        # 计算正确预测数
        correct = (preds == targets).sum().item()
        self.total_correct += correct
        self.total_samples += len(targets)
        
        # 保存预测和目标值
        self.predictions.extend(preds.numpy())
        self.targets.extend(targets.numpy())
        
    def compute(self):
        """
        计算所有指标
        
        返回:
            metrics: 指标字典
        """
        if self.total_samples == 0:
            return {
                'rank1_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
            
        # 计算准确率
        rank1_accuracy = self.total_correct / self.total_samples
        
        # 计算精确率、召回率和F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.targets,
            self.predictions,
            average='binary'
        )
        
        # 计算混淆矩阵
        cm = confusion_matrix(self.targets, self.predictions)
        
        # 计算每个类别的指标
        class_metrics = self._compute_class_metrics(cm)
        
        metrics = {
            'rank1_accuracy': rank1_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'class_metrics': class_metrics
        }
        
        return metrics
        
    def _compute_class_metrics(self, cm):
        """
        计算每个类别的指标
        
        参数:
            cm: 混淆矩阵
            
        返回:
            class_metrics: 每个类别的指标字典
        """
        num_classes = cm.shape[0]
        class_metrics = {}
        
        for i in range(num_classes):
            # 真阳性、假阳性、假阴性
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            # 计算精确率和召回率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # 计算F1分数
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[f'class_{i}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
        return class_metrics
        
    def get_confusion_matrix(self):
        """
        获取混淆矩阵
        
        返回:
            confusion_matrix: 混淆矩阵
        """
        return confusion_matrix(self.targets, self.predictions)
        
    def get_classification_report(self):
        """
        获取分类报告
        
        返回:
            report: 分类报告字符串
        """
        metrics = self.compute()
        
        report = "分类报告:\n"
        report += f"Rank-1准确率: {metrics['rank1_accuracy']:.4f}\n"
        report += f"精确率: {metrics['precision']:.4f}\n"
        report += f"召回率: {metrics['recall']:.4f}\n"
        report += f"F1分数: {metrics['f1']:.4f}\n\n"
        
        report += "每个类别的指标:\n"
        for class_name, class_metrics in metrics['class_metrics'].items():
            report += f"{class_name}:\n"
            report += f"  精确率: {class_metrics['precision']:.4f}\n"
            report += f"  召回率: {class_metrics['recall']:.4f}\n"
            report += f"  F1分数: {class_metrics['f1']:.4f}\n"
            
        return report 

def normalize_metrics(metrics: dict, count_key: str = 'total_count', exclude_keys=None) -> dict:
    """
    将metrics字典中的所有数值（除exclude_keys外）统一除以count_key指定的数量，返回新的均值字典。
    """
    if exclude_keys is None:
        exclude_keys = []
    count = metrics.get(count_key, 0)
    if count <= 0:
        return {k: 0.0 for k in metrics}
    normalized = {}
    for k, v in metrics.items():
        if k in exclude_keys:
            normalized[k] = v
        else:
            try:
                normalized[k] = float(v) / count
            except Exception:
                normalized[k] = 0.0
    return normalized 