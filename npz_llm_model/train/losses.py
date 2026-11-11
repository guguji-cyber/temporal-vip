"""
损失函数定义模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

# SentenceTransformer单例管理
class SentenceTransformerSingleton:
    _instances = {}

    @classmethod
    def get_instance(cls, model_path, device=None):
        if model_path not in cls._instances:
            cls._instances[model_path] = SentenceTransformer(model_path)
            if device and torch.cuda.is_available():
                cls._instances[model_path] = cls._instances[model_path].to(device)
        return cls._instances[model_path]
import os
from glob import glob


class TextSimilarityLoss(nn.Module):
    """文本相似度损失函数"""
    
    def __init__(self, weight=0.5, config=None):
        """
        初始化文本相似度损失
        
        参数:
            weight: 损失权重
        """
        super().__init__()
        self.weight = weight
        # 使用单例模式避免重复加载
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.model = SentenceTransformerSingleton.get_instance(config.st_path, current_device)
    
    def compute_similarity(self, predictions, targets, valid_mask=None):
        """
        计算文本相似度，考虑有效性掩码
        
        参数:
            predictions: 预测的文本列表
            targets: 目标文本列表
            valid_mask: 有效样本掩码，True表示有效样本
        """
        if valid_mask is None:
            valid_mask = torch.ones(len(predictions), dtype=torch.bool)
        
        # 只处理有效样本
        valid_predictions = [p for p, v in zip(predictions, valid_mask) if v]
        valid_targets = [t for t, v in zip(targets, valid_mask) if v]
        
        if not valid_predictions:
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                target_device = f'cuda:{current_device}'
            else:
                target_device = 'cpu'
            return torch.tensor([], device=target_device)
        
        # 获取文本嵌入
        pred_embeddings = self.model.encode(valid_predictions, convert_to_tensor=True)
        target_embeddings = self.model.encode(valid_targets, convert_to_tensor=True)

        # 确保嵌入在正确的设备上（使用当前设备）
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            target_device = f'cuda:{current_device}'
        else:
            target_device = 'cpu'
        pred_embeddings = pred_embeddings.to(target_device)
        target_embeddings = target_embeddings.to(target_device)

        # 计算余弦相似度
        similarities = util.pytorch_cos_sim(pred_embeddings, target_embeddings)
        diagonal_similarities = torch.diagonal(similarities)

        return diagonal_similarities
    
    def forward(self, pred_explanations, target_explanations, valid_mask=None):
        """
        计算文本相似度损失，考虑有效性掩码
        
        参数:
            pred_explanations: 预测的解释文本列表
            target_explanations: 目标解释文本列表
            valid_mask: 有效样本掩码，True表示有效样本（可为[B, N]或[B]）
        """
        # 自动处理掩码维度：若为二维，则每个batch只要有一个有效人物就算有效
        if valid_mask is not None and hasattr(valid_mask, 'ndim') and valid_mask.ndim == 2:
            valid_mask = valid_mask.any(dim=1)  # [B]
        # 计算相似度
        similarities = self.compute_similarity(pred_explanations, target_explanations, valid_mask)
        
        if similarities.size(0) == 0:
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                target_device = f'cuda:{current_device}'
            else:
                target_device = 'cpu'
            return torch.tensor(0.0, device=target_device), similarities
        
        # 计算损失：1 - 相似度
        loss = 1.0 - similarities.mean()
        
        return self.weight * loss, similarities


class ClassificationLoss(nn.Module):
    def __init__(self, weight=1.0, config=None):
        super().__init__()
        self.weight = weight
        self.config = config
    
    def forward(self, logits, targets, valid_mask=None):
        """
        参数:
            logits: [B, N] 预测的概率分布
            targets: [B] 重要人物的索引（全局索引）
            valid_mask: [B, N] 有效人物的掩码（True/False）
        返回:
            loss: 平均交叉熵损失，只在有效人物之间归一化
        """
        if valid_mask is not None:
            # 将无效人物的 logits 置为 -inf，使 softmax 后概率为0
            masked_logits = logits.masked_fill(~valid_mask, float('-inf'))
        else:
            masked_logits = logits
        # 直接批量计算 cross_entropy，softmax 只在有效人之间归一化
        loss = F.cross_entropy(masked_logits, targets, reduction='mean')
        return self.weight * loss


class ContrastiveLoss(nn.Module):
    """
    标准的InfoNCE (Info Noise Contrastive Estimation) 损失实现
    让重要人物的特征更相似，与非重要人物的特征更不相似
    """
    
    def __init__(self, temperature=0.5, weight=0.3, config=None):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.config = config
        
    def forward(self, features, targets, valid_mask=None):
        """
        计算InfoNCE损失
        
        参数:
            features: 特征向量，形状为[B, N, D]，B为batch size，N为每个样本中的人数，D为特征维度
            targets: 重要人物的索引，形状为[B]，表示每个样本中重要人物的索引位置
            valid_mask: 有效样本掩码，形状为[B]
        """
        device = features.device
        
        if valid_mask is None:
            valid_mask = torch.ones(features.size(0), dtype=torch.bool, device=device)
            
        # 只使用有效样本
        valid_features = features[valid_mask]  # [valid_B, N, D]
        valid_targets = targets[valid_mask]    # [valid_B]
        
        if valid_features.size(0) < 2:
            return torch.tensor(0.0, device=device)
            
        # 获取每个样本中重要人物的特征
        B, N, D = valid_features.shape
        anchor_features = valid_features[torch.arange(B, device=device), valid_targets]  # [B, D]
        
        # L2归一化
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        
        # 创建正样本和负样本
        positives = []  # 正样本：其他样本中的重要人物特征
        negatives = []  # 负样本：同一样本中的非重要人物特征
        
        for i in range(B):
            # 收集正样本：其他样本中的重要人物特征
            pos_indices = torch.arange(B, device=device) != i
            positives.append(anchor_features[pos_indices])  # [B-1, D]
            
            # 收集负样本：当前样本中的非重要人物特征
            # 创建一个与N相同大小的布尔掩码，标记非目标索引位置
            neg_mask = torch.ones(N, dtype=torch.bool, device=device)
            neg_mask[valid_targets[i]] = False
            curr_negatives = valid_features[i][neg_mask]  # [N-1, D]
            curr_negatives = F.normalize(curr_negatives, p=2, dim=1)
            negatives.append(curr_negatives)
            
        # 计算logits
        loss = torch.tensor(0.0, device=device)
        for i in range(B):
            anchor = anchor_features[i:i+1]  # [1, D]
            
            # 正样本相似度
            if len(positives[i]) > 0:
                pos_sim = torch.matmul(anchor, positives[i].T)  # [1, B-1]
                pos_sim = pos_sim / self.temperature
            
            # 负样本相似度
            if len(negatives[i]) > 0:
                neg_sim = torch.matmul(anchor, negatives[i].T)  # [1, N-1]
                neg_sim = neg_sim / self.temperature
                
                # 组合正负样本的相似度
                if len(positives[i]) > 0:
                    all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [1, (B-1)+(N-1)]
                    
                    # 创建标签：正样本的索引为0到B-2
                    labels = torch.zeros(1, dtype=torch.long, device=device)
                    
                    # 计算交叉熵损失
                    curr_loss = F.cross_entropy(all_sim, labels)
                    loss = loss + curr_loss
        
        # 平均每个样本的损失
        loss = loss / B if B > 0 else loss
        
        return self.weight * loss


class WeightRegularizationLoss(nn.Module):
    """权重正则化损失"""
    
    def __init__(self, weight=0.0001):
        """
        初始化权重正则化损失
        
        参数:
            weight: 正则化系数
        """
        super().__init__()
        self.weight = weight
        
    def forward(self, model):
        """
        计算L2正则化损失（所有参数都加正则，保证DDP兼容）
        """
        l2_reg = torch.tensor(0., device=next(model.parameters()).device)
        param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)
                param_count += 1
        if param_count > 0:
            l2_reg = l2_reg / param_count
        return self.weight * l2_reg


class RankingLoss(nn.Module):
    """Pairwise Margin Ranking Loss"""
    def __init__(self, margin=0.1, weight=1.0):
        super().__init__()
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        self.weight = weight
    def forward(self, score1, score2, target):
        # score1, score2: [num_pairs], target: [num_pairs] (1 or -1)
        if score1.numel() == 0:
            return torch.tensor(0.0, device=score1.device)
        return self.weight * self.loss_fn(score1, score2, target)


class TotalLoss(nn.Module):
    """总损失计算"""
    
    def __init__(self, cls_weight=1.0, text_weight=0.5, contrastive_weight=0.3, reg_weight=0.0001, ranking_weight=0.2, ranking_margin=0.1, config=None):
        """
        初始化总损失计算器
        
        参数:
            cls_weight: 分类损失权重
            text_weight: 文本相似度损失权重
            contrastive_weight: 对比损失权重
            reg_weight: 正则化损失权重
            ranking_weight: 排序损失权重
            ranking_margin: 排序损失的margin参数
        """
        super().__init__()
        self.classification_loss = ClassificationLoss(weight=cls_weight, config=config)
        self.text_similarity_loss = TextSimilarityLoss(weight=text_weight, config=config)
        self.contrastive_loss = ContrastiveLoss(weight=contrastive_weight, config=config)
        self.regularization_loss = WeightRegularizationLoss(weight=reg_weight)
        self.ranking_loss = RankingLoss(margin=ranking_margin, weight=ranking_weight)
        self.config = config
        
    def forward(self, model_outputs, targets, model):
        """
        计算总损失
        
        参数:
            model_outputs: 模型输出字典，包含：
                - logits: 分类logits
                - features: 特征向量 [B, N, D]
                - explanations: 生成的解释文本
            targets: 目标字典，包含：
                - indices: 目标索引 [B]，表示每个样本中重要人物的索引
                - explanations: 目标解释文本
                - valid_mask: 有效样本掩码
            model: 模型实例，用于计算正则化损失
        """
        
        # 获取有效样本掩码
        valid_mask = targets.get('valid_mask', None)
        # 分类损失用二维mask
        cls_loss = self.classification_loss(
            model_outputs['logits'], 
            targets['indices'],
            valid_mask
        )
        # 文本损失直接传原始mask，内部自动处理
        if hasattr(self.config, 'is_ablation') and self.config.is_ablation:
            text_loss = torch.tensor(0.0, device=cls_loss.device if hasattr(cls_loss, 'device') else 'cpu')
            similarities = None
        else:
            text_loss, similarities = self.text_similarity_loss(
                model_outputs['explanations'],
                targets['explanations'],
                valid_mask
            )
        # 计算对比损失：只保留重要人物有效的batch
        features = model_outputs['features']  # [B, N, D]
        if features is not None:
            if valid_mask is not None and valid_mask.ndim == 2:
                B = valid_mask.shape[0]
                indices = targets['indices']
                batch_valid = torch.tensor(
                    [valid_mask[i, indices[i]] for i in range(B)],
                    dtype=torch.bool, device=valid_mask.device
                )  # [B]
            else:
                batch_valid = valid_mask
            contrastive_loss = self.contrastive_loss(
                features,  # [B, N, D]
                targets['indices'],  # [B]
                batch_valid
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        reg_loss = self.regularization_loss(model)
        # DDP兼容：强制所有参数都被访问一次，dummy loss加到总loss
        dummy = torch.tensor(0.0, device=reg_loss.device)
        for name, param in model.named_parameters():
            if param.requires_grad:
                dummy = dummy + param.sum() * 0
        # 排序损失
        if 'ranking_scores' in model_outputs:
            score1, score2, ranking_target = model_outputs['ranking_scores']
            ranking_loss = self.ranking_loss(score1, score2, ranking_target)
        else:
            ranking_loss = torch.tensor(0.0, device=cls_loss.device if hasattr(cls_loss, 'device') else 'cpu')
        # 确保所有损失项在同一设备上（使用cls_loss的设备作为基准）
        target_device = cls_loss.device
        text_loss = text_loss.to(target_device)
        contrastive_loss = contrastive_loss.to(target_device)
        reg_loss = reg_loss.to(target_device)
        ranking_loss = ranking_loss.to(target_device)
        dummy = dummy.to(target_device)

        # 计算总损失
        total_loss = cls_loss + text_loss + contrastive_loss + reg_loss + ranking_loss + dummy
        # 返回损失明细和相似度
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'text_similarity_loss': text_loss,
            'contrastive_loss': contrastive_loss,
            'regularization_loss': reg_loss,
            'ranking_loss': ranking_loss,
            'text_similarities': similarities,
            'ranking_scores': model_outputs.get('ranking_scores', None)
        }