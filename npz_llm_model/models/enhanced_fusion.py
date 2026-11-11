"""
增强型多模态融合模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class AttentionPooling(nn.Module):
    """
    可学习的Attention Pooling，对T维做加权聚合
    输入: [B, T, N, D] -> 输出: [B, N, D]
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        q = self.query.expand(B, N, -1)  # [B, N, D]
        x_proj = self.proj(x)  # [B, T, N, D]
        attn = (x_proj * q.unsqueeze(1)).sum(-1)  # [B, T, N]
        attn = attn.softmax(dim=1)  # T维归一化
        out = (x * attn.unsqueeze(-1)).sum(1)  # [B, N, D]
        return out

class EnhancedMultiModalFusion(nn.Module):
    """
    多模态特征融合层（支持attention pooling时序建模）
    输入：
        static_features: [B, T, N, 3, D_s] 或 [B, N, D_s]
        dynamic_features: [B, T, N, 2, D_d] 或 [B, N, D_d]
        text_features: [B, N, D_t]
        person_masks: [B, N]
    输出：
        fused: [B, N, D_f]
        debug_info: 融合过程信息
    """
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.static_dim = config.static_extractor['feature_dim']
        self.dynamic_dim = config.dynamic_extractor['feature_dim']
        self.text_dim = config.text_extractor['feature_dim']
        self.fusion_dim = config.temporal_alignment['fusion_dim']
        self.dropout = config.temporal_alignment['dropout']
        self.norm = config.temporal_alignment['use_norm']
        # 分支聚合
        self.static_proj = nn.Linear(self.static_dim, self.fusion_dim)
        self.dynamic_proj = nn.Linear(self.dynamic_dim, self.fusion_dim)
        # Attention Pooling
        self.attn_pool = AttentionPooling(self.fusion_dim)
        # 融合MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, self.fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(self.dropout)
        )
        self.register_buffer("empty_features", torch.zeros(1, self.fusion_dim))

    def aggregate_k(self, features: torch.Tensor) -> torch.Tensor:
        """
        对K分支（如3/2）做mean聚合，输入[B, T, N, K, D]，输出[B, T, N, D]
        """
        if features is None:
            return None
        if features.dim() == 5:
            return features.mean(dim=3)
        return features  # 已聚合

    def forward(
        self,
        static_features: torch.Tensor = None,   # [B, T, N, 3, D_s] or [B, N, D_s]
        dynamic_features: torch.Tensor = None,  # [B, T, N, 2, D_d] or [B, N, D_d]
        text_features: torch.Tensor = None,     # [B, N, D_t]
        person_masks: torch.Tensor = None       # [B, N]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        1. 对K分支聚合: [B, T, N, K, D] -> [B, T, N, D]
        2. 静态/动态分别投影到fusion_dim
        3. Attention Pooling对T聚合，输出全局特征
        4. 三分支拼接+MLP融合
        """
        try:
            debug_info = {}
            # 1. K分支聚合
            static_agg = self.aggregate_k(static_features) if static_features is not None else None  # [B, T, N, D_s] or [B, N, D_s]
            dynamic_agg = self.aggregate_k(dynamic_features) if dynamic_features is not None else None  # [B, T, N, D_d] or [B, N, D_d]
            # 2. 投影
            if static_agg is not None:
                if static_agg.dim() == 3:
                    static_proj = self.static_proj(static_agg)  # [B, N, fusion_dim]
                else:
                    static_proj = self.static_proj(static_agg)  # [B, T, N, fusion_dim]
            else:
                static_proj = None
            if dynamic_agg is not None:
                if dynamic_agg.dim() == 3:
                    dynamic_proj = self.dynamic_proj(dynamic_agg)
                else:
                    dynamic_proj = self.dynamic_proj(dynamic_agg)
            else:
                dynamic_proj = None
            # 3. Attention Pooling对T聚合
            if static_proj is not None and static_proj.dim() == 4:
                static_global = self.attn_pool(static_proj)  # [B, N, fusion_dim]
            else:
                static_global = static_proj  # [B, N, fusion_dim] or None
            if dynamic_proj is not None and dynamic_proj.dim() == 4:
                dynamic_global = self.attn_pool(dynamic_proj)
            else:
                dynamic_global = dynamic_proj
            # 文本特征直接拼接
            text_global = text_features  # [B, N, D_t] or None
            # 4. 三分支拼接+MLP融合
            fusion_inputs = [f for f in [static_global, dynamic_global, text_global] if f is not None]
            fusion_input = torch.cat(fusion_inputs, dim=-1)  # [B, N, D_sum]
            debug_info["fusion_input"] = fusion_input.detach()
            # 动态调整MLP输入维度
            if fusion_input.shape[-1] != self.mlp[0].in_features:
                self.mlp[0] = nn.Linear(fusion_input.shape[-1], self.fusion_dim).to(fusion_input.device)
            fused = self.mlp(fusion_input)
            if fusion_input.shape[-1] == self.fusion_dim:
                fused = fused + fusion_input
            if person_masks is not None:
                person_masks_agg = person_masks.any(dim=1) if person_masks.dim() > 2 else person_masks  # [B, N]
                fused = fused * person_masks_agg.unsqueeze(-1)
            debug_info["fused"] = fused.detach()
            return fused, debug_info
        except Exception as e:
            if person_masks is not None and len(person_masks.shape) >= 2:
                B, N = person_masks.shape[:2]
            else:
                B, N = 1, 1
            return self.empty_features.expand(B, N, -1), {"error": str(e)} 