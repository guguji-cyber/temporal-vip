"""
静态特征提取模块 - 基于Transformer的多特征融合
整合画面占比、中心度、人脸清晰度等静态特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import mediapipe as mp
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class FeatureEncoder(nn.Module):
    """特征编码器：将标量特征扩展到高维空间"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, config.static_extractor['feature_encoder_hidden']),
            nn.ReLU(),
            nn.Linear(config.static_extractor['feature_encoder_hidden'], config.static_extractor['feature_dim']),
            nn.LayerNorm(config.static_extractor['feature_dim']),
            nn.Dropout(config.static_extractor['dropout'])
        )
    
    def forward(self, x):
        return self.net(x.unsqueeze(-1))

class TemporalAttention(nn.Module):
    """时序注意力：捕获特征的时序变化"""
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.static_extractor['feature_dim'],
            num_heads=config.static_extractor['temporal_heads'],
            batch_first=True,
            dropout=config.static_extractor['dropout']
        )
        self.norm = nn.LayerNorm(config.static_extractor['feature_dim'])
        
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        return self.norm(x + attn_out)

class StaticFeatureExtractor(nn.Module):
    """静态特征提取器"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frame_height = config.frame_height
        self.frame_width = config.frame_width
        self.feature_dim = config.static_extractor['feature_dim']
        
        # 视觉特征提取 - 使用ResNet18
        if config.static_extractor['backbone'] == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.visual_encoder = nn.Sequential(*list(resnet.children())[:-2])
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
                
        # 特征编码器
        self.feature_encoders = nn.ModuleDict({
            name: FeatureEncoder(config)
            for name in config.static_extractor['feature_names']
        })
        
        # 时序特征处理
        self.temporal_attention = TemporalAttention(config)
        
        # 特征融合
        self.cross_attention = nn.MultiheadAttention(
            self.feature_dim, 
            num_heads=config.static_extractor['temporal_heads'],
            batch_first=True,
            dropout=config.static_extractor['dropout']
        )
        
        # 特征投影
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(config.static_extractor['dropout'])
        )
        
        # Mediapipe初始化
        try:
            self.mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        except Exception as e:
            logger.error(f"Mediapipe初始化失败: {str(e)}")
            raise RuntimeError("Mediapipe初始化失败")
            
        self.register_buffer("empty_features", torch.zeros(1, len(config.static_extractor['feature_names']), self.feature_dim))
        
    def extract_face_clarity(self, frame: np.ndarray, bbox: np.ndarray) -> float:
        """提取人脸清晰度"""
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
                
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0
                
            results = self.mp_face.process(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
            if results.multi_face_landmarks:
                clarity = cv2.Laplacian(roi, cv2.CV_64F).var()
                return float(clarity) / 1000.0
                
        except Exception as e:
            logger.error(f"人脸清晰度计算失败: {str(e)}")
            
        return 0.0
        
    def compute_static_features(self, frames, bboxes, masks):
        """计算基础静态特征（area/centrality/clarity均为person级mask）"""
        B, T, H, W, C = frames.shape
        N = bboxes.shape[2]
        device = frames.device

        # area/centrality 矢量化
        boxes = bboxes  # [B, T, N, 4]
        x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        area = ((x2 - x1) * (y2 - y1)) / (W * H)
        area = torch.clamp(area, min=0)
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        center_dist = torch.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        centrality = 1.0 - torch.clamp(center_dist / 0.5, max=1.0)

        # area/centrality/clarity 统一归一化
        # area
        area_proc = torch.log1p(area)
        # centrality
        centrality_proc = torch.log1p(centrality)
        # clarity
        clarity = torch.zeros(B, T, N, device=device)
        for b in range(B):
            for t in range(T):
                for n in range(N):
                    if not masks[b, t, n]:
                        continue
                    box = bboxes[b, t, n].cpu().numpy()
                    frame_np = frames[b, t].cpu().numpy().astype(np.uint8)
                    clarity[b, t, n] = self.extract_face_clarity(frame_np, box)
        clarity_proc = torch.log1p(clarity)

        # 统一无效目标填充
        for feat_proc in [area_proc, centrality_proc, clarity_proc]:
            valid_mask = (masks > 0)
            if valid_mask.sum() > 0:
                mean_valid = feat_proc[valid_mask].mean()
            else:
                mean_valid = 0.0
            feat_proc[~valid_mask] = mean_valid

        # 统一clip到[0, 1.6]区间
        area_clipped = torch.clamp(area_proc, min=0.0, max=1.6)
        centrality_clipped = torch.clamp(centrality_proc, min=0.0, max=1.6)
        clarity_clipped = torch.clamp(clarity_proc, min=0.0, max=1.6)
        area_norm = area_clipped / 1.6
        centrality_norm = centrality_clipped / 1.6
        clarity_norm = clarity_clipped / 1.6

        features = {
            'area': area_norm,
            'centrality': centrality_norm,
            'clarity': clarity_norm
        }
        
        # 根据当前配置只返回需要的特征
        active_features = {name: features[name] for name in self.config.static_extractor['feature_names'] if name in features}
        return active_features
        
    @torch.amp.autocast('cuda')
    def forward(
        self,
        frames: torch.Tensor,
        bboxes: torch.Tensor,
        person_masks: torch.Tensor,
        frame_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """前向传播"""
        try:
            B, T, H, W, C = frames.shape
            N = bboxes.shape[2]
            device = frames.device
            
            # 1. 提取基础特征
            raw_features = self.compute_static_features(frames, bboxes, person_masks)
            
            # 2. 特征编码
            encoded_features = {}
            for feat_name, feat_values in raw_features.items():
                encoded = self.feature_encoders[feat_name](feat_values)  # [B, T, N, D]
                encoded_features[feat_name] = encoded
            
            # 3. 特征堆叠与聚合（去除复杂attention，仅做简单聚合）
            # 动态堆叠为 [B, T, N, num_features, D]
            feature_list = [encoded_features[name] for name in self.config.static_extractor['feature_names'] if name in encoded_features]
            stacked_features = torch.stack(feature_list, dim=3)  # [B, T, N, num_features, D]

            # 保留T维度，不做mean聚合
            static_features = self.projection(stacked_features)  # [B, T, N, num_features, D]

            # === 防御机制：确保每个序列至少有一个未被mask的token ===
            # person_valid: [B, N] (True为有效)
            # mask用于T维时序聚合/注意力时，需保证每行至少有一个False
            # 这里假设后续如有T维mask用到person_masks[:, :, n]，则加防御
            # 以person_masks为例，若有全False行，强制第一个为True
            if person_masks is not None:
                # [B, T, N] -> [B*N, T]
                mask_T = person_masks.permute(0, 2, 1).reshape(-1, person_masks.shape[1])
                all_false_rows = ~mask_T.any(dim=1)
                if all_false_rows.any():
                    mask_T[all_false_rows, 0] = True
                # 若后续有T维mask用mask_T，则用此防御后的mask_T

            # 应用mask
            person_valid = torch.any(person_masks, dim=1)
            static_features = static_features * person_valid.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

            # 准备调试信息
            debug_info = {
                "encoded_features": encoded_features,
                "stacked_features": stacked_features
            }
            
            # 动态构建 raw_features
            active_raw_features = {name: raw_features[name] for name in self.config.static_extractor['feature_names'] if name in raw_features}

            return static_features, active_raw_features, debug_info
            
        except Exception as e:
            import traceback
            logger.error(f"静态特征提取失败: {str(e)}\n{traceback.format_exc()}")
            # 确保返回的空张量具有正确的5D形状
            B = frames.shape[0] if 'frames' in locals() and frames is not None else 1
            T = frames.shape[1] if 'frames' in locals() and frames is not None else self.config.max_frames
            N = bboxes.shape[2] if 'bboxes' in locals() and bboxes is not None else self.config.max_persons
            num_features = len(self.config.static_extractor['feature_names'])
            
            empty_5d = torch.zeros(B, T, N, num_features, self.feature_dim, device=frames.device if 'frames' in locals() and frames is not None else 'cpu')
            return empty_5d, None, None 