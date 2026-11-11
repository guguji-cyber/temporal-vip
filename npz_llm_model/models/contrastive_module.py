"""
多粒度对比学习模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class MultiGranularityContrastive(nn.Module):
    """
    多粒度对比学习模块（与主流程标准一致）
    输入：
        scene_text: 场景文本特征 [B, D_t]
        person_text: 人物文本特征 [B, N, D_t]
        static_features: 静态特征 [B, N, 3, D_s]
        dynamic_features: 动态特征 [B, N, 2, D_d]
        person_ids: 人物ID [B, N]
        person_valid: 有效掩码 [B, N]
    """
    
    def __init__(self, config: Any):
        """
        初始化多粒度对比学习模块
        
        参数:
            config: 配置对象
        """
        # super(MultiGranularityContrastive, self).__init__()
        super().__init__()
        self.config = config
        
        # 温度参数调整对比学习锐度
        self.temperature = getattr(config, 'contrastive_temperature', 0.07)
        
        # 对齐空间的维度
        self.projection_dim = getattr(config, 'fusion_dim', 256)
        
        # 特征投影层，确保文本和视觉特征在同一空间
        self.text_dim = config.text_extractor['feature_dim']
        self.static_dim = config.static_extractor['feature_dim']
        self.dynamic_dim = config.dynamic_extractor['feature_dim']
        self.agg_method = getattr(config, 'fusion_agg_method', 'mean')
        self.text_projector = nn.Linear(self.text_dim, self.projection_dim)
        self.static_projector = nn.Linear(self.static_dim, self.projection_dim)
        self.dynamic_projector = nn.Linear(self.dynamic_dim, self.projection_dim)
        
        # 损失权重
        self.global_weight = getattr(config, 'global_weight', 0.5)
        self.local_weight = getattr(config, 'local_weight', 0.5)
        
    def aggregate(self, features: torch.Tensor, method: str) -> torch.Tensor:
        """
        特征聚合，支持mean/max
        """
        if method == 'mean':
            return features.mean(dim=2)
        elif method == 'max':
            return features.max(dim=2)[0]
        else:
            raise ValueError(f"不支持的聚合方式: {method}")

    def compute_similarity(self, text_feat, visual_feat):
        """
        计算文本和视觉特征的相似度
        
        参数:
            text_feat: 文本特征
            visual_feat: 视觉特征
            
        返回:
            similarity: 相似度矩阵
        """
        # 标准化特征
        text_feat = F.normalize(text_feat, dim=-1)
        visual_feat = F.normalize(visual_feat, dim=-1)
        
        # 计算相似度
        similarity = torch.matmul(text_feat, visual_feat.transpose(-2, -1))
        return similarity / self.temperature
        
    def forward(
        self,
        scene_text: torch.Tensor,         # [B, D_t]
        person_text: torch.Tensor,        # [B, N, D_t]
        static_features: torch.Tensor,    # [B, N, D_s]
        dynamic_features: torch.Tensor,   # [B, N, D_d]
        person_ids: torch.Tensor,         # [B, N]
        person_valid: torch.Tensor        # [B, N]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        前向传播计算多粒度对比学习损失
        输入特征已由主模型聚合，无需再对T维池化
        """
        batch_size, num_persons = static_features.shape[0], static_features.shape[1]
        device = scene_text.device
        
        if self.config and getattr(self.config, 'debug', False):
            print(f"[对比学习] 输入形状:")
            print(f"  scene_text: {scene_text.shape}")
            print(f"  person_text: {person_text.shape}")
            print(f"  static_features: {static_features.shape}")
            print(f"  person_valid: {person_valid.shape}")
            print(f"  有效人物数: {person_valid.sum().item()}")
        
        # 1. 投影
        scene_text_proj = self.text_projector(scene_text)  # [B, D]
        person_text_proj = self.text_projector(person_text)  # [B, N, D]
        static_proj = self.static_projector(static_features)  # [B, N, D]
        dynamic_proj = self.dynamic_projector(dynamic_features)  # [B, N, D]
        
        # 2. 全局对比
        valid_mask = person_valid.float().unsqueeze(-1)  # [B, N, 1]
        batch_valid = valid_mask.sum(dim=1).squeeze(-1) > 0  # [B]
        if not batch_valid.any():
            return torch.tensor(0.0, device=device), {
                'global_loss': 0.0,
                'local_loss': 0.0
            }
        global_visual = (static_proj + dynamic_proj) / 2  # [B, N, D]
        global_visual = (global_visual * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)  # [B, D]
        valid_scene_text = scene_text_proj[batch_valid]  # [V_B, D]
        valid_global_video = global_visual[batch_valid]  # [V_B, D]
        if len(valid_scene_text) > 0:
            global_sim = self.compute_similarity(valid_scene_text, valid_global_video)  # [V_B, V_B]
            global_labels = torch.arange(len(valid_scene_text), device=device)
            global_loss = F.cross_entropy(global_sim, global_labels)
        else:
            global_loss = torch.tensor(0.0, device=device)
        
        # 3. 局部对比
        person_visual = (static_proj + dynamic_proj) / 2  # [B, N, D]
        flat_person_text = person_text_proj.reshape(-1, person_text_proj.size(-1))  # [B*N, D]
        flat_person_visual = person_visual.reshape(-1, person_visual.size(-1))  # [B*N, D]
        flat_valid = person_valid.reshape(-1)  # [B*N]
        valid_count = flat_valid.sum().item()
        if valid_count == 0:
            return global_loss, {
                'global_loss': global_loss.item(),
                'local_loss': 0.0
            }
        valid_indices = torch.nonzero(flat_valid).squeeze(-1)  # [V]
        try:
            valid_text = flat_person_text[valid_indices]  # [V, D]
            valid_visual = flat_person_visual[valid_indices]  # [V, D]
            local_sim = self.compute_similarity(valid_text, valid_visual)  # [V, V]
            local_labels = torch.arange(len(valid_indices), device=device)
            local_loss = F.cross_entropy(local_sim, local_labels)
        except Exception as e:
            return global_loss, {
                'global_loss': global_loss.item(),
                'local_loss': 0.0
            }
        
        # 4. 总体损失
        total_loss = self.global_weight * global_loss + self.local_weight * local_loss
        
        return total_loss, {'global_loss': global_loss, 'local_loss': local_loss} 