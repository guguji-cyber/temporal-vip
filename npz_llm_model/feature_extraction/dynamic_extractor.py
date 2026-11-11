"""
动态特征提取模块 - 基于3D ResNet和Transformer的动作和唇动特征提取
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class BasicBlock3D(nn.Module):
    """标准3D ResNet残差块，自动处理shape变化"""
    def __init__(self, in_channels, out_channels, stride=(1,2,2)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if stride != (1,1,1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class Resnet3D(nn.Module):
    """3D ResNet用于动作特征提取（采用标准残差块）"""
    def __init__(self, config):
        super().__init__()
        self.in_channels = 3
        self.base_channels = 64
        self.num_layers = config.dynamic_extractor['action_encoder']['num_layers']
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.base_channels, kernel_size=(3,7,7), 
                     stride=(1,2,2), padding=(1,3,3), bias=False),
            nn.BatchNorm3d(self.base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )
        # ResNet块
        self.layers = nn.ModuleList()
        curr_channels = self.base_channels
        for i in range(self.num_layers):
            out_channels = curr_channels * 2
            stride = (1,2,2) if i > 0 else (1,1,1)
            self.layers.append(BasicBlock3D(curr_channels, out_channels, stride=stride))
            curr_channels = out_channels
        # 时序注意力
        self.temporal_attention = nn.MultiheadAttention(
            curr_channels,
            num_heads=config.dynamic_extractor['action_encoder']['temporal_heads'],
            batch_first=True,
            dropout=config.dynamic_extractor['action_encoder']['dropout']
        )
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(curr_channels, config.dynamic_extractor['feature_dim']),
            nn.LayerNorm(config.dynamic_extractor['feature_dim']),
            nn.Dropout(config.dynamic_extractor['action_encoder']['dropout'])
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for block in self.layers:
            x = block(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T, -1, C)
        x = x.mean(dim=2)
        x = x + self.temporal_attention(x, x, x)[0]
        return self.projection(x)

class LipEncoder(nn.Module):
    """唇动特征编码器"""
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.dynamic_extractor['lip_encoder']['input_dim']
        self.hidden_dim = config.dynamic_extractor['lip_encoder']['hidden_dim']
        self.num_layers = config.dynamic_extractor['lip_encoder']['num_layers']
        
        # 特征编码
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.dynamic_extractor['lip_encoder']['temporal_heads'],
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.dynamic_extractor['lip_encoder']['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, config.dynamic_extractor['feature_dim']),
            nn.LayerNorm(config.dynamic_extractor['feature_dim']),
            nn.Dropout(config.dynamic_extractor['lip_encoder']['dropout'])
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.projection(x)

class DynamicFeatureExtractor(nn.Module):
    """动态特征提取器"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.dynamic_extractor['feature_dim']

        # 唇动关键点缓存（纯加速，等价性已用 verify_lip_cache_strict.py 验证：
        # 三个 batch（2/3/4）上 action/lip/dynamic/logits 的 max|diff| 全为 0）。
        # 留空则走原始 MediaPipe 路径。
        from pathlib import Path as _Path
        _dirs = config.dynamic_extractor.get('lip_cache_dirs', []) or []
        self._lip_cache_dirs = [_Path(d) for d in _dirs if _Path(d).is_dir()]
        self._lip_index = None
        self._lip_hit = 0
        self._lip_miss = 0
        if self._lip_cache_dirs:
            logger.info(f"唇动缓存已启用: {[str(d) for d in self._lip_cache_dirs]}")

        # 动作特征提取
        self.action_encoder = Resnet3D(config)
        
        # 唇动特征提取
        self.lip_encoder = LipEncoder(config)

        # 可选：光流特征替换器（简单且轻量）
        if self.config.dynamic_extractor.get('replace_temporal_with_flow', False):
            class OpticalFlowEncoder(nn.Module):
                """非常简单的光流特征编码：对每帧计算相邻帧的光流(mean(u),mean(v))，线性投影到特征维度"""
                def __init__(self, cfg):
                    super().__init__()
                    self.feature_dim = cfg.dynamic_extractor['feature_dim']
                    self.dropout = cfg.dynamic_extractor['action_encoder']['dropout']
                    # 流图后处理小网络 (RAFT 输出或 Farneback 输出作为 2-channel 输入)
                    self.flow_cnn = nn.Sequential(
                        nn.Conv2d(2, 16, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(32, self.feature_dim),
                        nn.LayerNorm(self.feature_dim),
                        nn.Dropout(self.dropout)
                    )

                    # 后端选择：farneback 或 raft
                    of_conf = cfg.dynamic_extractor.get('optical_flow', {})
                    self.backend = of_conf.get('backend', 'farneback')
                    self.raft_model = None
                    self.raft_device = of_conf.get('raft_device', 'cpu')
                    self.raft_model_path = of_conf.get('raft_model_path', '')
                    # 如果选择 RAFT，尝试加载模型（若无法加载将在第一次推理时降级或抛出错误）
                    if self.backend == 'raft':
                        try:
                            import torch as _torch
                            # 如果未提供 checkpoint 路径，尝试自动下载默认raft_things
                            if not self.raft_model_path:
                                try:
                                    from npz_llm_model.utils.raft_utils import download_raft_checkpoint
                                    dl = download_raft_checkpoint('raft_things')
                                    if dl:
                                        self.raft_model_path = dl
                                except Exception:
                                    logger.warning('无法自动下载 RAFT checkpoint，若需使用请手动提供 --raft_model_path')
                            # 尝试通过 torch.hub 加载 raft_small/raft
                            try:
                                model_name = 'raft_small' if of_conf.get('raft_small', True) else 'raft'
                                self.raft_model = _torch.hub.load('princeton-vl/RAFT', model_name, pretrained=False)
                            except Exception:
                                # 尝试导入本地 raft 包中的 RAFT 类（如果用户已安装）
                                try:
                                    from raft import RAFT as _RAFT
                                    self.raft_model = _RAFT()
                                except Exception:
                                    self.raft_model = None
                            # 如给出 checkpoint 路径，尝试加载
                            if self.raft_model is not None and self.raft_model_path:
                                ckpt = _torch.load(self.raft_model_path, map_location='cpu')
                                sd = ckpt.get('state_dict', ckpt)
                                # 处理可能的 key 前缀
                                new_sd = {}
                                for k, v in sd.items():
                                    nk = k
                                    if k.startswith('module.'):
                                        nk = k[len('module.'):]
                                    new_sd[nk] = v
                                self.raft_model.load_state_dict(new_sd)
                                self.raft_model.to(self.raft_device)
                                self.raft_model.eval()
                        except Exception as e:
                            logger.error(f"尝试加载 RAFT 模型失败: {e}")
                            self.raft_model = None

                def forward(self, clips_uint8, person_valid_mask, frame_masks_per_clip):
                    """Compute per-frame features from optical flow using configured backend (RAFT or Farneback)

                    Args:
                        clips_uint8: tensor [num_valid, T, H, W, 3] (uint8) or numpy array
                        person_valid_mask: [num_valid, T] bool tensor
                        frame_masks_per_clip: [num_valid, T] bool tensor

                    Returns:
                        feats: tensor [num_valid, T, feature_dim]
                    """
                    # Ensure inputs are tensors on CPU for cv2 compatibility
                    if not isinstance(clips_uint8, torch.Tensor):
                        clips_uint8 = torch.from_numpy(clips_uint8)
                    clips_uint8 = clips_uint8.cpu()
                    person_valid_mask = person_valid_mask.cpu()
                    frame_masks_per_clip = frame_masks_per_clip.cpu()

                    num_valid, T, H, W, C = clips_uint8.shape
                    feats = torch.zeros(num_valid, T, self.feature_dim)

                    try:
                        for i in range(num_valid):
                            clip_np = clips_uint8[i].numpy()

                            # If RAFT backend selected and model available, try RAFT; otherwise use Farneback
                            if self.backend == 'raft' and self.raft_model is not None:
                                # prepare float tensors for RAFT
                                img_tensors = [torch.from_numpy(f.astype('float32') / 255.0).permute(2, 0, 1) for f in clip_np]

                                for t in range(T - 1):
                                    if not (person_valid_mask[i, t] and person_valid_mask[i, t + 1] and frame_masks_per_clip[i, t] and frame_masks_per_clip[i, t + 1]):
                                        continue

                                    prev = img_tensors[t].unsqueeze(0).to(self.raft_device)
                                    nxt = img_tensors[t + 1].unsqueeze(0).to(self.raft_device)

                                    # pad to multiples of 8
                                    ph = ((prev.shape[2] + 7) // 8) * 8
                                    pw = ((prev.shape[3] + 7) // 8) * 8
                                    pad_h = ph - prev.shape[2]
                                    pad_w = pw - prev.shape[3]
                                    prev_p = torch.nn.functional.pad(prev, (0, pad_w, 0, pad_h), mode='reflect')
                                    nxt_p = torch.nn.functional.pad(nxt, (0, pad_w, 0, pad_h), mode='reflect')

                                    with torch.no_grad():
                                        try:
                                            out = self.raft_model(prev_p, nxt_p, iters=20, test_mode=True)
                                            # RAFT often returns (flow_low, flow_up)
                                            if isinstance(out, (tuple, list)):
                                                flow = out[-1]
                                            else:
                                                flow = out
                                            flow = flow[0][:, :prev.shape[2], :prev.shape[3]]  # [2,H,W]
                                            flow_tensor = flow.cpu()
                                        except Exception as e:
                                            logger.error(f"RAFT 推理失败，回退到 Farneback: {e}")
                                            flow_tensor = None

                                    if flow_tensor is None:
                                        # fallback to Farneback when RAFT fails
                                        gr_prev = cv2.cvtColor(clip_np[t].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                                        gr_nxt = cv2.cvtColor(clip_np[t + 1].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                                        flow_np = cv2.calcOpticalFlowFarneback(gr_prev, gr_nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                        flow_tensor = torch.from_numpy(flow_np.transpose(2, 0, 1)).float()

                                    # run through small CNN to get feature vector
                                    flow_in = flow_tensor.unsqueeze(0)  # [1,2,H,W]
                                    # Move input to the same device and dtype as flow_cnn parameters to avoid dtype mismatch
                                    try:
                                        param = next(self.flow_cnn.parameters())
                                        flow_in = flow_in.to(device=param.device, dtype=param.dtype)
                                    except StopIteration:
                                        # no params available, keep as is
                                        pass
                                    except Exception:
                                        pass
                                    # Run the small CNN without autocast to avoid mixed-precision issues
                                    try:
                                        from torch.cuda import amp as _amp
                                        with _amp.autocast(enabled=False):
                                            feat = self.flow_cnn(flow_in).squeeze(0)
                                    except Exception:
                                        # fallback if amp unavailable
                                        feat = self.flow_cnn(flow_in).squeeze(0)
                                    # store as float32 on CPU to keep downstream consistent
                                    feats[i, t, :] = feat.detach().cpu().to(torch.float32)

                                # copy last frame feature
                                if T >= 2:
                                    feats[i, -1, :] = feats[i, -2, :]

                            else:
                                # Farneback baseline
                                grays = [cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in clip_np]
                                for t in range(T - 1):
                                    if not (person_valid_mask[i, t] and person_valid_mask[i, t + 1] and frame_masks_per_clip[i, t] and frame_masks_per_clip[i, t + 1]):
                                        continue
                                    prev = grays[t]
                                    nxt = grays[t + 1]
                                    flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                    flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
                                    flow_in = flow_tensor.unsqueeze(0)
                                    try:
                                        param = next(self.flow_cnn.parameters())
                                        flow_in = flow_in.to(device=param.device, dtype=param.dtype)
                                    except StopIteration:
                                        pass
                                    except Exception:
                                        pass
                                    try:
                                        from torch.cuda import amp as _amp
                                        with _amp.autocast(enabled=False):
                                            feat = self.flow_cnn(flow_in).squeeze(0)
                                    except Exception:
                                        feat = self.flow_cnn(flow_in).squeeze(0)
                                    feats[i, t, :] = feat.detach().cpu().to(torch.float32)

                                if T >= 2:
                                    feats[i, -1, :] = feats[i, -2, :]

                    except Exception as e:
                        logger.error(f"光流特征提取失败: {e}", exc_info=True)

                    return feats
            self.optical_flow_encoder = OpticalFlowEncoder(self.config)

            # If config.device indicates a CUDA device and raft_device is 'auto' or default 'cpu', align it
            try:
                cfg_dev = getattr(self.config, 'device', None)
                if hasattr(self.optical_flow_encoder, 'raft_device') and cfg_dev and isinstance(cfg_dev, str) and 'cuda' in cfg_dev:
                    if (not self.optical_flow_encoder.raft_device) or str(self.optical_flow_encoder.raft_device).startswith('cpu') or str(self.optical_flow_encoder.raft_device).lower() == 'auto':
                        self.optical_flow_encoder.raft_device = cfg_dev
                        if getattr(self.optical_flow_encoder, 'raft_model', None) is not None:
                            try:
                                self.optical_flow_encoder.raft_model.to(self.optical_flow_encoder.raft_device)
                            except Exception:
                                logger.warning(f"无法移动 RAFT 模型到设备 {self.optical_flow_encoder.raft_device}")
            except Exception:
                pass
        
        # 特征融合
        self.cross_attention = nn.MultiheadAttention(
            self.feature_dim,
            num_heads=config.dynamic_extractor['action_encoder']['temporal_heads'],
            batch_first=True,
            dropout=config.dynamic_extractor['action_encoder']['dropout']
        )
        
        # 特征投影
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(config.dynamic_extractor['action_encoder']['dropout'])
        )
        
        # Adjusted empty_features shape to [B, T, N, 2, D_f]
        # Use 0 for B, T, N to allow expansion later based on actual batch dimensions
        self.register_buffer("empty_features", 
                           torch.zeros(1, 1, 1, len(config.dynamic_extractor['feature_names']), self.feature_dim))
        
        try:
            self.mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        except Exception as e:
            logger.error(f"Mediapipe初始化失败: {str(e)}")
            self.mp_face = None
        
    @torch.amp.autocast('cuda')
    def extract_person_clips(
        self,
        frames: torch.Tensor,
        bboxes: torch.Tensor,
        person_masks: torch.Tensor,
        frame_masks: torch.Tensor
    ) -> torch.Tensor:
        """提取每个人物的时序片段，ImageNet均值/方差归一化"""
        B, T, H, W, C = frames.shape
        N = bboxes.shape[2]
        clips = torch.zeros(B, N, C, T, 32, 32, device=frames.device)
        try:
            for b in range(B):
                for n in range(N):
                    if not torch.any(person_masks[b, :, n]):
                        continue
                    for t in range(T):
                        if not person_masks[b, t, n]:
                            continue
                        box = bboxes[b, t, n].cpu().numpy()
                        frame_np = frames[b, t].cpu().numpy().astype(np.uint8)
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W-1, x2), min(H-1, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        roi = frame_np[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        # resize to 32x32
                        roi_resized = cv2.resize(roi, (32, 32))
                        clips[b, n, :, t] = torch.tensor(roi_resized.transpose(2, 0, 1), device=frames.device)
            # ImageNet均值/方差归一化
            mean = torch.tensor([0.485, 0.456, 0.406], device=clips.device).view(1, 3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=clips.device).view(1, 3, 1, 1, 1)
            clips = clips / 255.0
            clips = (clips - mean) / std
        except Exception as e:
            logger.error(f"提取人物片段时出错: {str(e)}", exc_info=True)
        return clips

    def extract_person_clips_uint8(
        self,
        frames: torch.Tensor,
        bboxes: torch.Tensor,
        person_masks: torch.Tensor,
        frame_masks: torch.Tensor
    ) -> torch.Tensor:
        """与 extract_person_clips 相同，但返回 uint8 原始 ROI 以供光流计算。

        返回形状: [B, N, T, 32, 32, 3] (dtype uint8)
        """
        B, T, H, W, C = frames.shape
        N = bboxes.shape[2]
        clips = np.zeros((B, N, T, 32, 32, 3), dtype=np.uint8)
        try:
            for b in range(B):
                for n in range(N):
                    if not torch.any(person_masks[b, :, n]):
                        continue
                    for t in range(T):
                        if not person_masks[b, t, n] or not frame_masks[b, t]:
                            continue
                        box = bboxes[b, t, n].cpu().numpy()
                        frame_np = frames[b, t].cpu().numpy().astype(np.uint8)
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W-1, x2), min(H-1, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        roi = frame_np[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        roi_resized = cv2.resize(roi, (32, 32))
                        clips[b, n, t] = roi_resized
        except Exception as e:
            logger.error(f"提取uint8人物片段时出错: {str(e)}", exc_info=True)
        # return torch tensor on cpu for easier cv2 usage later
        return torch.from_numpy(clips)

    @torch.amp.autocast('cuda')
    def extract_lip_sequence(
        self,
        frames: torch.Tensor,
        bboxes: torch.Tensor,
        person_mask: torch.Tensor,
        frame_mask: torch.Tensor
    ) -> torch.Tensor:
        """提取唇动关键点序列。

        若配置了 lip_cache_dirs（由 baselines/precompute_lip.py 生成），则直接
        返回缓存中的关键点序列，跳过 MediaPipe。缓存内容就是本函数的原始返回值，
        等价性已由 baselines/verify_lip_cache_strict.py 在多个 batch 上验证：
        action / lip / dynamic / 模型 logits 的 max|diff| 全为 0。

        缓存键：video_id + person index。这两个值由 forward 在调用前临时写到
        self._lip_ctx，用完即清；未设置时走原始 MediaPipe 路径。
        """
        # ---------- 缓存命中 ----------
        ctx = getattr(self, "_lip_ctx", None)
        if ctx is not None and self._lip_cache_dirs:
            vid, n = ctx
            if self._lip_index is None:
                idx = {}
                for d in self._lip_cache_dirs:
                    for f in d.glob("*.npz"):
                        idx[f.stem] = f
                self._lip_index = idx
            path = self._lip_index.get(str(vid))
            if path is not None:
                with np.load(path) as z:
                    key = f"lip_{n}"
                    self._lip_hit += 1
                    if key in z:
                        return torch.from_numpy(z[key]).float().to(frames.device)
                    # 缓存里没有该人物 => 原始实现会返回全零序列
                    T_, _ = frames.shape[0], None
                    return torch.zeros(frames.shape[0], 40, device=frames.device)
            self._lip_miss += 1

        # ---------- 原始 MediaPipe 路径（未改动） ----------
        T, H, W, C = frames.shape
        lip_seq = torch.zeros(T, 40, device=frames.device)
        try:
            for t in range(T):
                if not person_mask[t] or not frame_mask[t]:
                    continue
                box = bboxes[t].cpu().numpy()
                frame_np = frames[t].cpu().numpy().astype(np.uint8)
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W-1, x2), min(H-1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = frame_np[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                results = self.mp_face.process(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                if results and results.multi_face_landmarks:
                    lips = [results.multi_face_landmarks[0].landmark[i] for i in range(61, 81)]
                    lips_xy = np.array([[p.x, p.y] for p in lips]).flatten()
                    if lips_xy.shape[0] == 40:
                        lips_xy = np.clip(lips_xy, 0, 1)
                        lip_seq[t] = torch.tensor(lips_xy, device=frames.device)
        except Exception as e:
            logger.error(f"提取唇动序列时出错: {str(e)}", exc_info=True)
            return torch.zeros_like(lip_seq)
        return lip_seq

    @torch.amp.autocast('cuda')
    def forward(
        self, 
        frames: torch.Tensor, 
        bboxes: torch.Tensor, 
        person_masks: torch.Tensor, 
        frame_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        前向传播
        
        提取和融合动作与唇动特征。
        
        Args:
            frames (Tensor): 视频帧序列 [B, T, H, W, 3]
            bboxes (Tensor): 人物边界框 [B, T, N, 4]
            person_masks (Tensor): 人物有效性掩码 [B, T, N]
            frame_masks (Tensor): 帧有效性掩码 [B, T]
            
        Returns:
            Tuple[Tensor, Dict, Dict]: 
                - dynamic_features: 动态特征 [B, T, N, 2, D_f] # Adjusted return shape
                - raw_features: 原始特征
                - debug_info: 调试信息
        """
        try:
            B, T, H, W, C = frames.shape
            N = bboxes.shape[2]
            device = frames.device
            
            # 1. 提取基础特征 (人物片段)
            clips = self.extract_person_clips(frames, bboxes, person_masks, frame_masks)
            # 人物有效性与索引（供两条路径使用）
            person_valid_BN = torch.any(person_masks, dim=1)
            valid_indices_BN = torch.nonzero(person_valid_BN, as_tuple=False)
            valid_clips = clips[person_valid_BN]
            action = torch.zeros(B, T, N, self.feature_dim, device=device)
            # 如果启用光流替换：提取uint8片段并得到光流特征
            use_flow = self.config.dynamic_extractor.get('replace_temporal_with_flow', False)
            if use_flow:
                clips_uint8 = self.extract_person_clips_uint8(frames, bboxes, person_masks, frame_masks)
                # 按有效 (b,n) 提取 uint8 clips 并计算光流特征
                # clips_uint8 在 CPU 上（np->tensor），确保索引 mask 在 CPU 上以避免device不匹配
                mask_cpu = person_valid_BN.cpu()
                valid_clips_uint8 = clips_uint8[mask_cpu].to('cpu') if clips_uint8.numel() > 0 else torch.zeros(0)
                if valid_clips_uint8.numel() > 0:
                    # person_valid_mask for each valid (b,n)
                    valid_person_masks = []
                    frame_masks_per_clip = []
                    for (b, n) in valid_indices_BN:
                        valid_person_masks.append(person_masks[b].to('cpu')[:, n])
                        frame_masks_per_clip.append(frame_masks[b].to('cpu'))
                    valid_person_masks = torch.stack(valid_person_masks).to(torch.bool)
                    frame_masks_per_clip = torch.stack(frame_masks_per_clip).to(torch.bool)
                    flow_feats_valid = self.optical_flow_encoder(valid_clips_uint8, valid_person_masks, frame_masks_per_clip)
                else:
                    flow_feats_valid = torch.zeros(0, T, self.feature_dim, device=device)
                # 将光流特征同时填充到 action 和 lip（替换两者）
                if flow_feats_valid.numel() > 0:
                    for idx, (b, n) in enumerate(valid_indices_BN):
                        action[b, :, n, :] = flow_feats_valid[idx].to(device)
                # lip 也用同样的光流特征
                lip = torch.zeros(B, T, N, self.feature_dim, device=device)
                if flow_feats_valid.numel() > 0:
                    for idx, (b, n) in enumerate(valid_indices_BN):
                        lip[b, :, n, :] = flow_feats_valid[idx].to(device)
            else:
                # NOTE: 此前这里的 else 误挂在冗余的内层 `if use_flow:` 上，
                # 导致 use_flow=False（默认/论文配置）时 action_encoder 永不执行、
                # action 恒为全零。现已对齐到外层分支。
                if valid_clips.numel() > 0:
                    action_feats_valid = self.action_encoder(valid_clips)
                else:
                    action_feats_valid = torch.zeros(0, T, self.feature_dim, device=device)
                if action_feats_valid.numel() > 0:
                    for idx, (b, n) in enumerate(valid_indices_BN):
                        action[b, :, n, :] = action_feats_valid[idx]
            # 如果使用光流替换，则唇动已在上面填充；否则按原流程提取唇动特征
            if not use_flow:
                lip = torch.zeros(B, T, N, self.feature_dim, device=device)
                valid_lip_indices_BN = []
                valid_lip_sequences = []
                # 若启用唇动缓存，设置 (video_id, person_idx) 上下文供
                # extract_lip_sequence 查表；不启用时 _lip_ctx 为 None，走原始路径。
                _vids = getattr(self, "_lip_video_ids", None)
                for b in range(B):
                    for n in range(N):
                        if person_valid_BN[b, n]:
                            self._lip_ctx = (_vids[b], n) if _vids is not None else None
                            lip_seq = self.extract_lip_sequence(
                                frames[b], bboxes[b, :, n], person_masks[b, :, n], frame_masks[b])
                            if torch.any(lip_seq != 0):
                                valid_lip_indices_BN.append((b, n))
                                valid_lip_sequences.append(lip_seq)
                self._lip_ctx = None
                if valid_lip_sequences:
                    valid_lip_sequences_stacked = torch.stack(valid_lip_sequences)
                    valid_lip_masks = [frame_masks[b] & person_masks[b, :, n] for (b, n) in valid_lip_indices_BN]
                    if valid_lip_masks:
                        valid_lip_masks_stacked = torch.stack(valid_lip_masks)
                        lip_encoder_mask = ~valid_lip_masks_stacked
                        all_true_rows = lip_encoder_mask.all(dim=1)
                        if all_true_rows.any():
                            lip_encoder_mask[all_true_rows, 0] = False
                    else:
                        lip_encoder_mask = None
                    projected_valid = self.lip_encoder(valid_lip_sequences_stacked, mask=lip_encoder_mask)
                else:
                    projected_valid = torch.zeros(0, T, self.feature_dim, device=device)
                if projected_valid.numel() > 0:
                    for idx, (b, n) in enumerate(valid_lip_indices_BN):
                        lip[b, :, n, :] = projected_valid[idx]
            dynamic_features = torch.stack([action, lip], dim=-2)
            # 按 feature_names 选择实际生效的 cue 通道，使 --ablate_feature 能够真正
            # 移除某个动态子流（此前固定 stack([action, lip]) 会忽略该配置）。
            # 注意：'speech' 是 'lip' 的配置别名（见 configs/config.py），两者等价。
            _names = list(self.config.dynamic_extractor.get('feature_names', ['action', 'speech'])) \
                if self.config is not None else ['action', 'speech']
            _alias = {'speech': 'lip', 'lip': 'lip', 'action': 'action',
                      'optical_flow': None}
            _keep = [False, False]  # 顺序对应 [action, lip]
            for _nm in _names:
                _key = _alias.get(str(_nm).lower())
                if _key == 'action':
                    _keep[0] = True
                elif _key == 'lip':
                    _keep[1] = True
                elif _key is None:
                    # 'optical_flow'：光流模式下 action/lip 均已被填为光流特征
                    _keep = [True, True]
                    break
            if not any(_keep):
                # 极端情形（例如两个 cue 都被消融）：保留全零占位，避免下游维度坍塌
                _keep = [True, True]
            if not all(_keep):
                dynamic_features = dynamic_features[..., [i for i, k in enumerate(_keep) if k], :]
            mask_expanded = person_masks.unsqueeze(-1).unsqueeze(-1).expand_as(dynamic_features)
            dynamic_features = dynamic_features.masked_fill(~mask_expanded, 0.0)
            norm_mask = torch.sum(dynamic_features, dim=-1, keepdim=True) != 0
           
            debug_info = {
                "action": action.detach().cpu(),
                "lip": lip.detach().cpu()
            }
            raw_features = {
                "action": action.detach(),
                "lip": lip.detach()
            }
            if self.config and getattr(self.config, 'debug', False):
                logger.debug(f"动作特征形状 : {action.shape}")
                logger.debug(f"唇动特征形状 : {lip.shape}")
                logger.debug(f"最终动态特征形状 : {dynamic_features.shape}")
            return dynamic_features, raw_features, debug_info
        except Exception as e:
            logger.error(f"动态特征提取失败: {str(e)}", exc_info=True)
            return self.empty_features.expand(B, T, N, -1, -1).to(device), None, None