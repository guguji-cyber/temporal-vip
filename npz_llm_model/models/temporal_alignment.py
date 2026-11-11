"""
时空感知的多模态特征融合模块（主流Cross-Attention+MLP+残差范式）
支持主流的特征聚合方式和交互机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Dict, Any, Optional, Union
import os

logger = logging.getLogger(__name__)

class NaNBatchException(Exception):
    pass

class AttentionPooling(nn.Module):
    """
    可学习的Attention Pooling，对T维做加权聚合
    输入: [..., T, D] -> 输出: [..., D]
    Mask: [..., T] (True for masked elements)
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: [..., T, D]
        # mask: [..., T] (True for masked elements)
        if x is None:
            return None

        prefix_shape = x.shape[:-2]
        T, D = x.shape[-2:]

        flat_x = x.view(-1, T, D)
        flat_B = flat_x.shape[0]

        q = self.query.expand(flat_B, 1, -1)

        x_proj = self.proj(flat_x)

        attention_scores = (x_proj * q).sum(-1)  # [flat_B, T]

        if mask is not None:
            flat_mask = mask.view(flat_B, T)
            # 检查是否全mask
            all_masked = flat_mask.all(dim=1)
            if all_masked.any():
                # 对于全mask的样本，直接返回全0
                out = torch.zeros(flat_B, D, device=flat_x.device, dtype=flat_x.dtype)
                # 只对未全mask的样本做正常pooling
                valid_idx = (~all_masked).nonzero(as_tuple=True)[0]
                if valid_idx.numel() > 0:
                    valid_x = flat_x[valid_idx]
                    valid_mask = flat_mask[valid_idx]
                    valid_x_proj = x_proj[valid_idx]
                    valid_q = q[valid_idx]
                    valid_attention_scores = (valid_x_proj * valid_q).sum(-1)
                    valid_attention_scores = valid_attention_scores.masked_fill(valid_mask, float("-inf"))
                    valid_attention_weights = torch.sigmoid(valid_attention_scores)
                    # nan/inf防御
                    valid_attention_weights = torch.nan_to_num(valid_attention_weights, nan=0.0, posinf=1.0, neginf=0.0)
                    valid_out = (valid_x * valid_attention_weights.unsqueeze(-1)).sum(dim=1)
                    valid_out = valid_out.to(out.dtype)  # 保证类型一致
                    out[valid_idx] = valid_out
                out = out.view(*prefix_shape, D)
                return out
            else:
                attention_scores = attention_scores.masked_fill(flat_mask, float("-inf"))

        attention_weights = torch.sigmoid(attention_scores)
        # nan/inf防御
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=1.0, neginf=0.0)
        out = (flat_x * attention_weights.unsqueeze(-1)).sum(dim=1)
        out = out.view(*prefix_shape, D)
        return out


class SubFeatureAggregator(nn.Module):
    """
    子特征聚合器，对K维做加权聚合并提取权重
    输入: [..., K, D] -> 输出: ([..., D], [..., K])
    Mask: [..., K] (True for masked elements) - Not applicable for K features here?
    """

    def __init__(self, input_dim: int, num_features: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        # 使用Attention机制更合理：query可以是可学习参数，与K个子特征的D维向量做点积
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.proj = nn.Linear(
            input_dim, input_dim
        )  # Linear proj for values (optional, but common in attention)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [..., K, D]
        if x is None:
            return None, None  # Return None if input is None

        prefix_shape = x.shape[:-2]
        K, D = x.shape[-2:]

        flat_x = x.view(-1, K, D)  # [Batch*Other, K, D]
        flat_B = flat_x.shape[0]

        q = self.query.expand(flat_B, 1, -1)  # [Batch*Other, 1, D]
        x_proj = self.proj(flat_x)  # [Batch*Other, K, D]

        # 计算Attention Score: [Batch*Other, K, D] * [Batch*Other, 1, D] -> [Batch*Other, K]
        # 对query和key进行点积作为score
        attention_scores = (x_proj * q).sum(-1)  # Dot product attention

        # 对K维应用Softmax，得到加和为1的权重: [Batch*Other, K] -> [Batch*Other, K]
        # attention_weights = torch.softmax(attention_scores, dim=-1)
        # 改为sigmoid激活，每个分支权重独立[0,1]
        attention_weights = torch.sigmoid(attention_scores)

        # 使用权重对K维特征进行加权求和: [Batch*Other, K, D] * [Batch*Other, K, 1] -> [Batch*Other, D]
        aggregated_feature = (flat_x * attention_weights.unsqueeze(-1)).sum(dim=1)

        # 重塑回原始前缀维度 + D 和原始前缀维度 + K
        aggregated_feature = aggregated_feature.view(*prefix_shape, D)
        attention_weights = attention_weights.view(*prefix_shape, K)  # 权重也重塑

        return aggregated_feature, attention_weights


class TemporalAwareAlignment(nn.Module):
    """
    多模态时序对齐+三分支融合（支持多种主流融合方式，可配置）
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.static_dim = config.static_extractor["feature_dim"]
        self.dynamic_dim = config.dynamic_extractor["feature_dim"]
        self.text_dim = config.text_extractor["feature_dim"]
        temporal_cfg = config.temporal_alignment
        self.fusion_dim = temporal_cfg["fusion_dim"]
        self.num_heads = temporal_cfg["num_heads"]
        self.num_layers = temporal_cfg["num_layers"]
        self.dropout = temporal_cfg["dropout"]
        self.norm = temporal_cfg["use_norm"]
        self.agg_method = temporal_cfg["agg_method"]
        self.cross_attention_layers = temporal_cfg["num_layers"]
        self.use_residual = temporal_cfg["use_residual"]
        self.gate_temperature = temporal_cfg["gate_temperature"]
        self.fusion_type = temporal_cfg.get("fusion_type", "transformer")
        # QKV mode: which modality provides Query in cross-attention
        # 'static_query' (default): Query from static (spatial), Key/Value from dynamic (temporal)
        # 'dynamic_query': Query from dynamic (temporal), Key/Value from static (spatial)
        self.qkv_mode = temporal_cfg.get('qkv_mode', 'static_query')

        # === 动态获取分支数 ===
        self.num_static_features = len(config.static_extractor["feature_names"])
        self.num_dynamic_features = len(config.dynamic_extractor["feature_names"])
        # 假设模态最多3种: static, dynamic, text
        self.num_modalities = len(config.feature_selection)

        self.static_sub_aggregator = SubFeatureAggregator(
            self.static_dim, self.num_static_features
        )
        self.dynamic_sub_aggregator = SubFeatureAggregator(
            self.dynamic_dim, self.num_dynamic_features
        )

        self.static_proj = nn.Linear(self.static_dim, self.fusion_dim)
        self.dynamic_proj = nn.Linear(self.dynamic_dim, self.fusion_dim)
        self.text_proj = nn.Linear(self.text_dim, self.fusion_dim)

        self.attn_pool_t = AttentionPooling(self.fusion_dim)

        if self.fusion_type in ["mlp", "gated", "transformer"]:
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=self.fusion_dim,
                        num_heads=self.num_heads,
                        batch_first=True,
                        dropout=self.dropout,
                    )
                    for _ in range(self.cross_attention_layers)
                ]
            )
            self.attn_pool_aligned = AttentionPooling(self.fusion_dim)

        if self.norm:
            self.layer_norm = nn.LayerNorm(self.fusion_dim)

        self.drop = nn.Dropout(self.dropout)

        if self.fusion_type == "concat":
            # Concat模式下，输入维度只由视觉模态（静态、动态）决定
            num_visual_modalities = ("static" in config.feature_selection) + ("dynamic" in config.feature_selection)
            # 保证维度不为0，以防没有选择任何视觉模态
            concat_input_dim = max(1, num_visual_modalities) * self.fusion_dim
            
            self.final_concat_mlp = nn.Sequential(
                nn.Linear(concat_input_dim, self.fusion_dim),
                nn.ReLU(),
                nn.LayerNorm(self.fusion_dim) if self.norm else nn.Identity(),
                nn.Dropout(self.dropout),
            )
        elif self.fusion_type == "mlp":
            self.final_mlp_fusion = nn.Sequential(
                nn.Linear(self.num_modalities * self.fusion_dim, self.fusion_dim),
                nn.ReLU(),
                nn.LayerNorm(self.fusion_dim) if self.norm else nn.Identity(),
                nn.Dropout(self.dropout),
                nn.Linear(self.fusion_dim, self.fusion_dim),
            )
        elif self.fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(self.num_modalities * self.fusion_dim, self.num_modalities),
                nn.Sigmoid(),
            )
        elif self.fusion_type == "transformer":
            modality_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.fusion_dim,
                nhead=self.num_heads,
                dim_feedforward=self.fusion_dim * 4,
                dropout=self.dropout,
                batch_first=True,
            )
            self.modality_transformer_fusion = nn.TransformerEncoder(
                modality_encoder_layer,
                num_layers=1,
            )

        self.register_buffer("empty_features", torch.zeros(1, self.fusion_dim))
        self.text_modulation_net = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )

    @torch.amp.autocast("cuda")
    def forward(
        self,
        static_features: Optional[torch.Tensor] = None,  # [B, T, N, num_static, D_s]
        dynamic_features: Optional[torch.Tensor] = None,  # [B, T, N, num_dynamic, D_d]
        text_features: Optional[torch.Tensor] = None,  # [B, N, D_t]（可选）
        person_masks: torch.Tensor = None,  # [B, N] (True for valid persons)
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        多模态时序对齐与融合
        1. 对静态/动态特征进行K分支聚合，提取加和为1的子特征权重
        2. 静态/动态/文本特征投影到fusion_dim
        3. 处理维度不一致 (扩展T和N维度) 到 [B, T, N, D_f]
        4. Cross-Attention对齐 (仅mlp/gated/transformer模式, 对齐静态和动态时序特征)
        5. 对齐后的特征进行T聚合 (Attention Pooling)
        6. 多分支融合 (根据fusion_type分支: concat/mlp/gated/transformer)
        7. 应用人物掩码
        """
        debug_log_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "vip_debug.txt"
        )

        prefix = "[train]" if self.training else "[val]"
        debug_info = {}
        device = (
            person_masks.device
            if person_masks is not None
            else (
                static_features.device
                if static_features is not None
                else (
                    dynamic_features.device
                    if dynamic_features is not None
                    else (text_features.device if text_features is not None else "cpu")
                )
            )
        )
        # 确定 B, T, N 的大小，根据 person_masks 的实际形状 [B, T, N] 进行提取
        if person_masks is not None:
            # 直接从 person_masks 的形状中提取 B, T, N
            if person_masks.dim() == 3:
                B, T, N = person_masks.shape
                # 可选：在这里添加检查 mask 的 T 维是否与特征的 T 维一致的断言或错误处理
                if (
                    static_features is not None and static_features.shape[1] != T
                ):  # static_features: [B, T, N, 3, D]
                    raise ValueError(
                        f"Person mask T dimension ({T}) does not match static feature T dimension ({static_features.shape[1]})"
                    )
                if (
                    dynamic_features is not None and dynamic_features.shape[1] != T
                ):  # dynamic_features: [B, T, N, 2, D]
                    raise ValueError(
                        f"Person mask T dimension ({T}) does not match dynamic feature T dimension ({dynamic_features.shape[1]})"
                    )
            else:
                # 如果不是三维，说明输入不符合预期，抛出错误
                raise ValueError(
                    f"Unsupported person_masks dimension: {person_masks.dim()}, expected 3 for [B, T, N]"
                )
        else:  # person_masks is None
            # 如果没有提供 person_masks，使用默认值
            B, N = 1, 1  # Default B and N
            T = 1  # Default T to 1
            if static_features is not None:
                T = static_features.shape[1]
            elif dynamic_features is not None:
                T = dynamic_features.shape[1]

        D_s_in = self.static_dim
        D_d_in = self.dynamic_dim
        D_t_in = self.text_dim
        D_f = self.fusion_dim

        # 1. 对K分支聚合，提取子特征权重
        # 静态特征: [B, T, N, 3, D_s_in]
        # if static_features is not None:
        #     shape = static_features.shape
        #     static_features = static_features.view(-1, shape[-2], shape[-1])
        #     static_features = torch.nn.LayerNorm([shape[-2], shape[-1]]).to(static_features.device)(static_features)
        #     static_features = static_features.view(*shape)
        aggregated_static_features, static_sub_weights = self.static_sub_aggregator(static_features)
        debug_info["static_sub_weights"] = (
            static_sub_weights.detach().cpu()
            if static_sub_weights is not None
            else None
        )

        # 计算静态子特征批次平均贡献度 (考虑 person_masks)
        if static_sub_weights is not None and person_masks is not None:
            num_static_features = static_features.shape[-2]
            static_sub_weights_batch_avg = torch.zeros(num_static_features, device=device)
            # Expand person_masks to [B, T, N, 1] for element-wise multiplication
            mask_expanded_static = person_masks.unsqueeze(-1)
            # Mask the weights: set weights of invalid persons/time steps to 0
            masked_weights_static = static_sub_weights * mask_expanded_static
            # Sum over B, T, N dimensions
            sum_weights_static = torch.sum(masked_weights_static, dim=(0, 1, 2))
            # Count valid elements (sum of mask)
            num_valid_elements_static = torch.sum(mask_expanded_static)
            # Calculate average, avoid division by zero
            if num_valid_elements_static > 0:
                sum_weights_per_subfeature = torch.sum(
                    masked_weights_static, dim=(0, 1, 2)
                )
                total_valid_person_time_steps = torch.sum(person_masks)
                if total_valid_person_time_steps > 0:
                    static_sub_weights_batch_avg = (
                        sum_weights_per_subfeature / total_valid_person_time_steps
                    )
                else:
                    static_sub_weights_batch_avg = torch.zeros(self.num_static_features, device=device)
        else:
            static_sub_weights_batch_avg = torch.zeros(self.num_static_features, device=device)
        debug_info["static_sub_weights_batch_avg"] = (
            static_sub_weights_batch_avg.detach().cpu()
        )
        aggregated_dynamic_features, dynamic_sub_weights = self.dynamic_sub_aggregator(dynamic_features)
        debug_info["dynamic_sub_weights"] = (
            dynamic_sub_weights.detach().cpu()
            if dynamic_sub_weights is not None
            else None
        )

        # 计算动态子特征批次平均贡献度 (考虑 person_masks)
        if dynamic_sub_weights is not None and person_masks is not None:
            num_dynamic_features = dynamic_features.shape[-2]
            dynamic_sub_weights_batch_avg = torch.zeros(num_dynamic_features, device=device)
            # Expand person_masks to [B, T, N, 1]
            mask_expanded_dynamic = person_masks.unsqueeze(-1)
            masked_weights_dynamic = dynamic_sub_weights * mask_expanded_dynamic
            sum_weights_per_subfeature = torch.sum(
                masked_weights_dynamic, dim=(0, 1, 2)
            )
            total_valid_person_time_steps = torch.sum(person_masks)
            if total_valid_person_time_steps > 0:
                dynamic_sub_weights_batch_avg = (
                    sum_weights_per_subfeature / total_valid_person_time_steps
                )
            else:
                dynamic_sub_weights_batch_avg = torch.zeros(self.num_dynamic_features, device=device)
        else:
            dynamic_sub_weights_batch_avg = torch.zeros(self.num_dynamic_features, device=device)
        debug_info["dynamic_sub_weights_batch_avg"] = (
            dynamic_sub_weights_batch_avg.detach().cpu()
        )

        # 2. 静态/动态/文本特征投影到fusion_dim
        # aggregated_static_features: [B, T, N, D_s_in] -> static_proj: [B, T, N, D_f]
        static_proj = (
            self.static_proj(aggregated_static_features)
            if aggregated_static_features is not None
            else None
        )
        # aggregated_dynamic_features: [B, T, N, D_d_in] -> dynamic_proj: [B, T, N, D_f]
        dynamic_proj = (
            self.dynamic_proj(aggregated_dynamic_features)
            if aggregated_dynamic_features is not None
            else None
        )
        # text_features: [B, N, D_t_in] -> text_proj: [B, N, D_f]
        text_proj = self.text_proj(text_features) if text_features is not None else None
        debug_info.update(
            {
                "static_proj": (
                    static_proj.detach().cpu() if static_proj is not None else None
                ),
                "dynamic_proj": (
                    dynamic_proj.detach().cpu() if dynamic_proj is not None else None
                ),
                "text_proj": (
                    text_proj.detach().cpu() if text_proj is not None else None
                ),
            }
        )

        # 3. 处理维度不一致 (扩展T和N维度) 到 [B, T, N, D_f]
        # 静态特征: static_proj 已经是 [B, T, N, D_f]
        static_expanded = static_proj
        # 动态特征: dynamic_proj [B, T, N, D_f]
        dynamic_expanded = dynamic_proj  # Assuming it's already [B, T, N, D_f]

        # 文本特征: text_proj [B, N, D_f], 需要扩展T维度
        if text_proj is not None:
            # 扩展T维度: [B, N, D_f] -> [B, 1, N, D_f] -> [B, T, N, D_f]
            text_expanded = text_proj.unsqueeze(1).expand(-1, T, -1, -1)
        else:
            text_expanded = None
        debug_info["text_expanded"] = (
            text_expanded.detach().cpu() if text_expanded is not None else None
        )

        # 所有模态特征现在理论上维度一致: [B, T, N, D_f]
        # 根据feature_selection选择用于后续处理的模态
        # 注意：这里的 fusion_modalities_expanded 是为了方便Cross-Attention和后续T聚合
        fusion_modalities_expanded = []
        modalities_present_expanded = []  # Keep track of which modalities are present
        if static_expanded is not None:
            fusion_modalities_expanded.append(static_expanded)
            modalities_present_expanded.append("static")
        if dynamic_expanded is not None:
            fusion_modalities_expanded.append(dynamic_expanded)
            modalities_present_expanded.append("dynamic")
        if text_expanded is not None:
            fusion_modalities_expanded.append(text_expanded)
            modalities_present_expanded.append("text")

        # 如果没有任何模态特征，返回空特征
        if not fusion_modalities_expanded:
            return self.empty_features.expand(B, N, -1).to(device), debug_info

        # === 新增: 文本调制视觉特征 (仅计算和应用调制，暂不改变后续流程输入) ===
        text_modulation_signal = None
        modulated_static_expanded = static_expanded
        modulated_dynamic_expanded = dynamic_expanded

        if text_expanded is not None:
            # 生成调制信号 [B, T, N, D_f]
            # 注意：text_global形状是 [B, N, D_f]，text_expanded形状是 [B, T, N, D_f]
            # 我们使用text_expanded (包含T维度复制) 来生成调制信号
            # 将text_expanded reshape到 [B*T*N, D_f] 输入MLP
            # 然后reshape回 [B, T, N, D_f]
            # 采用 [B*T*N, D_f] 输入，因为text_modulation_net是Sequential MLP
            # Check if B*T*N is greater than 0 before reshaping
            if B * T * N > 0:
                text_expanded_flat = text_expanded.reshape(-1, D_f)
                text_modulation_signal_flat = self.text_modulation_net(
                    text_expanded_flat
                )
                text_modulation_signal = text_modulation_signal_flat.reshape(
                    B, T, N, D_f
                )
            else:
                # If B*T*N is 0, cannot create modulation signal
                text_modulation_signal = None

            # 应用调制（乘法调制 + Sigmoid）
            if text_modulation_signal is not None:
                if modulated_static_expanded is not None:
                    # Ensure both tensors are on the same device before operation
                    modulated_static_expanded = (
                        modulated_static_expanded
                        * torch.sigmoid(
                            text_modulation_signal.to(modulated_static_expanded.device)
                        )
                    )
                if modulated_dynamic_expanded is not None:
                    # Ensure both tensors are on the same device before operation
                    modulated_dynamic_expanded = (
                        modulated_dynamic_expanded
                        * torch.sigmoid(
                            text_modulation_signal.to(modulated_dynamic_expanded.device)
                        )
                    )

            debug_info["text_modulation_signal"] = (
                text_modulation_signal.detach().cpu()
                if text_modulation_signal is not None
                else None
            )
            debug_info["modulated_static_expanded"] = (
                modulated_static_expanded.detach().cpu()
                if modulated_static_expanded is not None
                else None
            )
            debug_info["modulated_dynamic_expanded"] = (
                modulated_dynamic_expanded.detach().cpu()
                if modulated_dynamic_expanded is not None
                else None
            )

        # === 原有: Cross-Attention对齐 (仅mlp/gated/transformer模式, 对齐静态和动态时序特征) ===
        # 对齐后，希望得到一个更能代表时序交互的特征 [B, T, N, D_f]
        # 只在 static 和 dynamic 都存在且 fusion_type 匹配时进行 Cross-Attention 对齐
        # Note: Currently uses original static_expanded and dynamic_expanded. Will modify later to use modulated ones.
        aligned_features = None
        
        if (
            self.fusion_type in ["mlp", "gated", "transformer"]
            and modulated_static_expanded is not None
            and modulated_dynamic_expanded is not None
        ):
            # 为了Cross-Attention，需要将 [B, T, N, D_f] 重塑为 [B*N, T, D_f]
            # 注意使用调制后的特征
            static_attn_input = modulated_static_expanded.permute(0, 2, 1, 3).reshape(
                B * N, T, D_f
            )
            dynamic_attn_input = modulated_dynamic_expanded.permute(0, 2, 1, 3).reshape(
                B * N, T, D_f
            )

            # 根据配置选择 Q/K/V 来源
            if self.qkv_mode == 'static_query':
                x = static_attn_input  # Query 来自静态（空间）
                attn_key = dynamic_attn_input
                attn_value = dynamic_attn_input
            else:
                x = dynamic_attn_input  # Query 来自动态（时间）
                attn_key = static_attn_input
                attn_value = static_attn_input
            debug_info['qkv_mode'] = self.qkv_mode

            # 使用人物掩码来mask Cross-Attention中的无效时序步
            # person_masks: [B, N]
            # 需要一个 [B*N, T] 的 mask for key_padding_mask
            # 从 person_masks [B, N] 扩展到 [B, N, T]，再重塑到 [B*N, T]
            # person_masks: [B, T, N] (True for valid persons)
            # Cross-Attention mask 期望 [B*N, T] (True for masked elements)
            # 正确做法：先permute到[B, N, T]，再reshape成[B*N, T]
            attn_mask = (~person_masks).permute(0, 2, 1).reshape(B * N, T)
            # 主流防御性处理：防止全mask导致nan
            all_true_rows = attn_mask.sum(dim=1) == attn_mask.shape[1]
            if all_true_rows.any():
                attn_mask[all_true_rows, 0] = False  # 保证每行至少有一个有效token

            for attn in self.cross_attentions:
                # Cross-Attention: Query(static), Key/Value(dynamic)
                # src_key_padding_mask 用于 mask Key/Value 中的 padding token (这里是无效时序步)
                # query_padding_mask 也可以用于 mask Query 中的无效时序步
                # 我们可以对 query 也应用相同的 mask
                attn_output, _ = attn(
                    query=x,
                    key=attn_key,
                    value=attn_value,
                    key_padding_mask=attn_mask,
                )

                # 添加LayerNorm和Dropout
                # LayerNorm 应该应用于残差连接之前或之后，这里放在残差之后
                attn_output = self.drop(attn_output)

                # 残差连接
                # 确保残差连接维度匹配 [B*N, T, D_f] + [B*N, T, D_f] = [B*N, T, D_f]
                x = x + attn_output
                # LayerNorm放在残差连接后
                if self.norm:
                    # nan/inf防御
                    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
                    x = self.layer_norm(x)

            # 对齐后的特征 x 维度是 [B*N, T, D_f]
            # 重塑回 [B, T, N, D_f]
            aligned_features = x.reshape(B, N, T, D_f).permute(
                0, 2, 1, 3
            )  # Permute back to [B, T, N, D_f]
        debug_info["aligned_features"] = (
            aligned_features.detach().cpu() if aligned_features is not None else None
        )

        # T 聚合 Mask: person_masks [B, T, N]
        # AttentionPooling 期望 mask [..., T] (True for masked elements)
        # 我们需要一个形状为 [B, N, T] 的掩码，然后重塑为 [B*N, T]

        # 原始 person_masks 是 [B, T, N]。我们需要转换为 [B, N, T]。
        # person_masks 可能是 bool 类型，需要确保后续操作兼容
        if person_masks is not None:
            # Permute person_masks from [B, T, N] to [B, N, T]
            t_mask_for_attn_pooling = person_masks.permute(0, 2, 1)  # Shape [B, N, T]

            # AttentionPooling 期望 True for masked elements，person_masks 是 True for valid persons
            # 所以需要取反
            t_agg_mask_attn_pooling = (
                ~t_mask_for_attn_pooling
            )  # True for masked elements [B, N, T]

            # 重塑为 [B*N, T] 以匹配 AttentionPooling 的输入格式
            t_agg_mask_flat = t_agg_mask_attn_pooling.reshape(B * N, T)
        fused_visual_global = None  # Initialize fused_visual_global here

        # 计算模态批次平均贡献度 (仅 gated 模式相关，非 gated 模式为零)
        modality_weights_batch_avg = torch.zeros(
            self.num_modalities, device=device
        )

        if self.fusion_type == "concat":
            # Concat模式：分别T聚合调制后的静态和动态，然后拼接MLP
            static_global_modulated = None
            if modulated_static_expanded is not None:
                static_expanded_flat = modulated_static_expanded.permute(
                    0, 2, 1, 3
                ).reshape(B * N, T, D_f)
                if B * N > 0:
                    static_global_modulated = self.attn_pool_t(
                        static_expanded_flat, mask=t_agg_mask_flat
                    )  # AttentionPooling expects [..., T, D] and [..., T] mask
                if static_global_modulated is not None:
                    static_global_modulated = static_global_modulated.view(B, N, D_f)

            dynamic_global_modulated = None
            if modulated_dynamic_expanded is not None:
                dynamic_expanded_flat = modulated_dynamic_expanded.permute(
                    0, 2, 1, 3
                ).reshape(B * N, T, D_f)
                if B * N > 0:
                    dynamic_global_modulated = self.attn_pool_t(
                        dynamic_expanded_flat, mask=t_agg_mask_flat
                    )
                if dynamic_global_modulated is not None:
                    dynamic_global_modulated = dynamic_global_modulated.view(B, N, D_f)

            # 动态地将所有可用的视觉特征拼接起来
            visual_features_to_concat = []
            if static_global_modulated is not None:
                visual_features_to_concat.append(static_global_modulated)
            if dynamic_global_modulated is not None:
                visual_features_to_concat.append(dynamic_global_modulated)

            if visual_features_to_concat:
                concatenated_global = torch.cat(visual_features_to_concat, dim=-1)
                fused_visual_global = self.final_concat_mlp(concatenated_global)
            else:
                fused_visual_global = None

        elif self.fusion_type in ["mlp", "gated", "transformer"]:
            # 这些模式需要 Cross-Attention 后的特征以及调制后的原始时序特征
            # 确保 aligned_features, modulated_static_expanded, modulated_dynamic_expanded 存在
            # 如果 Cross-Attention 没有发生 (例如只有一种模态)，aligned_features 可能为None
            # 这种情况下，这些模式的输入需要调整

            fusion_inputs = []
            if aligned_features is not None:
                fusion_inputs.append(aligned_features)
            # 添加调制后的原始时序特征
            if modulated_static_expanded is not None:
                fusion_inputs.append(modulated_static_expanded)
            if modulated_dynamic_expanded is not None:
                fusion_inputs.append(modulated_dynamic_expanded)

            if not fusion_inputs:
                # 如果没有任何有效的视觉时序特征 (Cross-Attention结果和调制后的原始特征)，则无法进行这些模式的融合
                fused_visual_global = None
            else:
                # Stack the input features: list of [B, T, N, D_f] -> [B, T, N, NumInputs, D_f]
                if len(fusion_inputs) == 1:
                    # 单模态时无需stack，直接unsqueeze保持兼容
                    stacked_features = fusion_inputs[0].unsqueeze(-2)  # [B, T, N, 1, D_f]
                else:
                    stacked_features = torch.stack(
                        fusion_inputs, dim=-2
                    )  # Stack along a new dimension before D_f
                    # stack_shape = [B, T, N, len(fusion_inputs), D_f]

                # Ensure B*T*N > 0 before reshaping
                if B * T * N > 0:
                    flat_stacked_features = stacked_features.reshape(
                        B * T * N, len(fusion_inputs), D_f
                    )
                else:
                    flat_stacked_features = None  # Cannot reshape if size is 0
                    fused_visual_global = None  # Cannot compute fused features

                if flat_stacked_features is not None:
                    # 在flat_stacked_features相关断言前后加入如下保护（以gated分支为例，其他分支同理）：
                    # 1. 在flat_stacked_features生成后，先用nan_to_num清理
                    flat_stacked_features = torch.nan_to_num(flat_stacked_features, nan=0.0, posinf=1e6, neginf=-1e6)
                    # 2. 检查是否仍有NaN/Inf，如果有则raise特殊异常
                    if torch.isnan(flat_stacked_features).any() or torch.isinf(flat_stacked_features).any():
                        print('[保护] flat_stacked_features 出现NaN/Inf，跳过本batch')
                        raise NaNBatchException('flat_stacked_features has NaN/Inf')
                    # 3. 其余聚合/归一化操作分母加1e-6
                    # 例如：
                    # avg_gate_weights = sum_gate_weights / (total_valid_person_time_steps + 1e-6)
                    # 4. 其余分支如有类似断言也做同样处理

                    if self.fusion_type == "mlp":
                        # Reshape to [B*T*N, len(fusion_inputs) * D_f] for MLP
                        mlp_input = flat_stacked_features.reshape(
                            B * T * N, len(fusion_inputs) * D_f
                        )
                        fused_temp_features_flat = self.final_mlp_fusion(mlp_input)
                        # Reshape back to [B, T, N, D_f]
                        fused_temp_features = fused_temp_features_flat.reshape(
                            B, T, N, D_f
                        )
                        # T Aggregate the MLP output
                        fused_temp_features_pool_input = fused_temp_features.permute(
                            0, 2, 1, 3
                        ).reshape(B * N, T, D_f)
                        if B * N > 0:
                            fused_visual_global = self.attn_pool_t(
                                fused_temp_features_pool_input, mask=t_agg_mask_flat
                            )
                        else:
                            fused_visual_global = None
                        if fused_visual_global is not None:
                            fused_visual_global = fused_visual_global.view(B, N, D_f)

                    elif self.fusion_type == "gated":
                        device = next(self.gate.parameters()).device
                        flat_stacked_features = flat_stacked_features.to(device).float()
                        # === 检查 shape ===
                        assert (
                            flat_stacked_features.dim() == 3
                        ), f"Expected 3D tensor, got {flat_stacked_features.shape}"
                        assert (
                            flat_stacked_features.shape[1] >= 1
                        ), f"Expected modality dim>=1, got {flat_stacked_features.shape[1]}"
                        assert (
                            flat_stacked_features.shape[0] > 0
                        ), f"B*T*N=0, got {flat_stacked_features.shape}"
                        assert not torch.isnan(
                            flat_stacked_features
                        ).any(), "flat_stacked_features has NaN"
                        assert not torch.isinf(
                            flat_stacked_features
                        ).any(), "flat_stacked_features has Inf"
                        # Apply gating on the stacked features [B*T*N, NumInputs, D_f]
                        gate_input = flat_stacked_features.reshape(
                            B * T * N, len(fusion_inputs) * D_f
                        )
                        # Gate output weights [B*T*N, NumInputs]
                        gate_weights = self.gate(gate_input)
                        gate_weights = torch.sigmoid(gate_weights)
                        # 修正：只统计有效mask下的门控权重
                        mask_flat = person_masks.reshape(-1, 1)  # [B*T*N, 1]
                        masked_gate_weights = gate_weights * mask_flat  # [B*T*N, NumInputs]
                        sum_gate_weights = masked_gate_weights.sum(dim=0)  # [NumInputs]
                        total_valid_person_time_steps = mask_flat.sum()  # 标量
                        if total_valid_person_time_steps > 0:
                            avg_gate_weights = sum_gate_weights / (total_valid_person_time_steps + 1e-6)
                            # 使用动态的模态数
                            modality_weights_batch_avg = torch.zeros(len(fusion_inputs), device=device)
                            for i in range(len(fusion_inputs)):
                                modality_weights_batch_avg[i] = avg_gate_weights[i]
                        else:
                            # 使用动态的模态数
                            modality_weights_batch_avg = torch.zeros(len(fusion_inputs), device=device)
                        debug_info["modality_weights_batch_avg"] = (
                            modality_weights_batch_avg.detach().cpu()
                        )
                        # Apply weights: [B*T*N, NumInputs, D_f] * [B*T*N, NumInputs, 1] -> [B*T*N, NumInputs, D_f]
                        weighted_features = (
                            flat_stacked_features * gate_weights.unsqueeze(-1)
                        )
                        # Sum along the modality dimension (dim 1, size NumInputs) -> [B*T*N, D_f]
                        fused_temp_features_flat = weighted_features.sum(dim=1)
                        fused_temp_features = fused_temp_features_flat.reshape(
                            B, T, N, D_f
                        )
                        fused_temp_features_pool_input = fused_temp_features.permute(
                            0, 2, 1, 3
                        ).reshape(B * N, T, D_f)
                        if B * N > 0:
                            fused_visual_global = self.attn_pool_t(
                                fused_temp_features_pool_input, mask=t_agg_mask_flat
                            )
                        else:
                            fused_visual_global = None
                        if fused_visual_global is not None:
                            fused_visual_global = fused_visual_global.view(B, N, D_f)

                    elif self.fusion_type == "transformer":
                        device = next(
                            self.modality_transformer_fusion.parameters()
                        ).device
                        flat_stacked_features = flat_stacked_features.to(device).float()
                        assert (
                            flat_stacked_features.dim() == 3
                        ), f"Expected 3D tensor, got {flat_stacked_features.shape}"
                        assert (
                            flat_stacked_features.shape[1] >= 1
                        ), f"Expected modality dim>=1, got {flat_stacked_features.shape[1]}"
                        assert (
                            flat_stacked_features.shape[0] > 0
                        ), f"B*T*N=0, got {flat_stacked_features.shape}"
                        assert not torch.isnan(
                            flat_stacked_features
                        ).any(), "flat_stacked_features has NaN"
                        assert not torch.isinf(
                            flat_stacked_features
                        ).any(), "flat_stacked_features has Inf"
                        transformer_output_flat = self.modality_transformer_fusion(
                            flat_stacked_features
                        )
                        # transformer_output_flat shape: [B*T*N, NumInputs, D_f]
                        fused_temp_features_flat = transformer_output_flat.mean(
                            dim=1
                        )  # [B*T*N, D_f]
                        fused_temp_features = fused_temp_features_flat.reshape(
                            B, T, N, D_f
                        )
                        fused_temp_features_pool_input = fused_temp_features.permute(
                            0, 2, 1, 3
                        ).reshape(B * N, T, D_f)
                        if B * N > 0:
                            fused_visual_global = self.attn_pool_t(
                                fused_temp_features_pool_input, mask=t_agg_mask_flat
                            )
                        else:
                            fused_visual_global = None
                        if fused_visual_global is not None:
                            fused_visual_global = fused_visual_global.view(B, N, D_f)

        # If no modalities were processed or B*N was 0, fused_visual_global might still be None
        # Handle the case where fused_visual_global is None (e.g., no visual inputs)
        if fused_visual_global is None:
            if B * N > 0:
                return self.empty_features.expand(B, N, -1).to(device), debug_info
        # 不要提前return empty_features，继续后续掩码处理

        # Save final visual global feature before masking
        debug_info["fused_visual_global_pre_mask"] = fused_visual_global.detach().cpu()

        # 7. 应用人物掩码
        # person_masks: [B, T, N]. fused_visual_global: [B, N, D_f]
        # 我们需要一个 [B, N] 的掩码来掩盖无效的人物
        if person_masks is not None and fused_visual_global is not None:
            # 从 [B, T, N] 的 person_masks 获取 [B, N] 的有效人物掩码 (True for valid persons)
            person_valid_mask_BN = torch.any(person_masks, dim=1)  # Shape [B, N]

            # 将 [B, N] 掩码扩展到 [B, N, D_f] 以匹配 fused_visual_global
            mask_expanded = person_valid_mask_BN.unsqueeze(-1).expand_as(
                fused_visual_global
            )

            # 使用布尔掩码将无效人物的特征清零 (True 表示有效，所以需要取反 ~)
            fused_visual_global = fused_visual_global.masked_fill(~mask_expanded, 0.0)

        # If fused_visual_global is not assigned (e.g., no input features), return empty features
        if fused_visual_global is None:
            return (
                self.empty_features.expand(B, N, -1).to(device),
                debug_info,
            )  # Ensure empty_features is on correct device and expanded to B, N

        # fused_visual_global有效mask下的均值、最大、最小
        if fused_visual_global is not None and person_masks is not None:
            valid_mask = torch.any(person_masks, dim=1)  # [B, N]
            valid_features = fused_visual_global[valid_mask]

        return fused_visual_global, debug_info
