"""
增强型Transformer模型，整合多粒度时空感知融合框架
- 支持三路特征提取（静态/动态/文本）
- 支持多粒度对比学习
- 支持时空感知对齐
- 支持增强型特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Any, Dict, Tuple, Optional, List, Union

from .base_model import BaseModel
from .contrastive_module import MultiGranularityContrastive
from .temporal_alignment import TemporalAwareAlignment, NaNBatchException

from ..feature_extraction.static_extractor import StaticFeatureExtractor
from ..feature_extraction.dynamic_extractor import DynamicFeatureExtractor
from ..feature_extraction.text_extractor import TextFeatureExtractor
from ..text_processing.explanation_generator import ExplanationGenerator
logger = logging.getLogger(__name__)

# 安全索引函数
def safe_index(tensor, *indices):
    assert tensor is not None, "tensor为None"
    assert tensor.dim() >= len(indices), f"tensor shape异常: {tensor.shape}, indices={indices}"
    return tensor[indices]

class EnhancedTransformerModel(BaseModel):
    """
    增强型Transformer模型（主流多模态框架）
    特征提取:
        - StaticFeatureExtractor: [B, N, 3, D_s] 空间特征
        - DynamicFeatureExtractor: [B, N, 2, D_d] 时序特征
        - TextFeatureExtractor: [B, N, D_t] 语义特征
    特征融合:
        - MultiGranularityContrastive: 多粒度对比学习
        - TemporalAwareAlignment: 时空感知对齐
        - EnhancedMultiModalFusion: 增强型特征融合
    """
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.config = config
        self.debug = getattr(config, 'debug', False)
        
        def _get_feature_selection(cfg):
            fs = getattr(cfg, 'feature_selection', ['static', 'dynamic', 'text'])
            if isinstance(fs, str):
                return [x.strip() for x in fs.split(',')]
            return fs
        feature_selection = _get_feature_selection(config)
        self.static_dim = config.static_feature_dim
        self.dynamic_dim = config.dynamic_feature_dim
        self.text_dim = config.text_feature_dim
        self.hidden_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else 256
        
        # 特征提取模块
        self.static_extractor = StaticFeatureExtractor(config)
        self.dynamic_extractor = DynamicFeatureExtractor(config)
        self.text_extractor = TextFeatureExtractor(config)
        # 增强模块
        self.contrastive_module = MultiGranularityContrastive(config)
        self.temporal_alignment = TemporalAwareAlignment(config)
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        # 分类层
        self.classifier = nn.Linear(config.hidden_dim, 1)

        # 解释生成器（只创建一次，避免重复加载模型）
        self.explanation_generator = None
        if not getattr(config, 'is_ablation', False):
            try:
                self.explanation_generator = ExplanationGenerator(config)
                # 设置为不需要梯度，因为解释生成器有自己的训练逻辑
                for param in self.explanation_generator.parameters():
                    param.requires_grad = False
                logger.info("解释生成器初始化成功")
            except Exception as e:
                logger.warning(f"解释生成器初始化失败: {e}")
                self.explanation_generator = None

        # CLIP相关配置
        config.clip_dim = getattr(config, 'clip_dim', 512)
        config.global_weight = getattr(config, 'global_weight', 0.5)
        config.local_weight = getattr(config, 'local_weight', 0.5)
        self._get_feature_selection = _get_feature_selection
    
    def _validate_inputs(self, batch_dict: Dict[str, Any]) -> None:
        """验证输入数据的有效性"""
        required_keys = ['frames', 'bboxes', 'person_masks', 'frame_masks',
                        'context_descriptions', 'person_descriptions', 'original_ids']
        for key in required_keys:
            assert key in batch_dict, f"缺少必要的输入: {key}"
        
        if self.debug:
            B = batch_dict['frames'].size(0)
            assert batch_dict['original_ids'].size(0) == B, "batch size不一致"
            assert batch_dict['person_masks'].dim() == 3, f"person_masks维度错误: {batch_dict['person_masks'].shape}"
    
    def _extract_features(
        self, 
        batch_dict: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """特征提取流程，按需提取"""
        static_features = dynamic_features = text_features = None
        static_debug = dynamic_debug = text_debug = None
        raw_static = raw_dynamic = None
        if 'static' in self.config.feature_selection:
            static_features, raw_static, static_debug = self.static_extractor(
                batch_dict['frames'],
                batch_dict['bboxes'],
                batch_dict['person_masks'],
                batch_dict['frame_masks']
            )
        if 'dynamic' in self.config.feature_selection:
            # 仅在启用唇动缓存时传入 video_ids 供查表；否则保持原始调用不变。
            if getattr(self.dynamic_extractor, '_lip_cache_dirs', None):
                self.dynamic_extractor._lip_video_ids = batch_dict.get('video_ids')
            dynamic_features, raw_dynamic, dynamic_debug = self.dynamic_extractor(
                batch_dict['frames'],
                batch_dict['bboxes'],
                batch_dict['person_masks'],
                batch_dict['frame_masks']
            )
            if hasattr(self.dynamic_extractor, '_lip_video_ids'):
                self.dynamic_extractor._lip_video_ids = None
        if 'text' in self.config.feature_selection:
            text_features, text_debug = self.text_extractor(
                batch_dict['context_descriptions'],
                batch_dict['person_descriptions'],
                batch_dict['original_ids']
            )
        debug_info = {
            'static': static_debug,
            'dynamic': dynamic_debug,
            'text': text_debug
        }
        return static_features, dynamic_features, text_features, debug_info, raw_static, raw_dynamic
    
    def generate_explanation(self, *args, **kwargs):
        # 使用已初始化的解释生成器，避免重复创建
        if self.explanation_generator is not None:
            return self.explanation_generator(*args, **kwargs)
        else:
            # 如果解释生成器未初始化，返回空结果
            logger.warning("解释生成器未初始化，返回空结果")
            return {
                'explanations': [],
                'losses': {'bart': [], 'semantic': [], 'feature': [], 'total': []},
                'video_info': [],
                'explanation_info': None
            }

    @torch.amp.autocast('cuda')
    def forward(
        self,
        batch_dict: Dict[str, Any],
        return_explanations: bool = True,
        return_attention: bool = False,
        criterion = None
    ) -> Dict[str, Any]:
        """
        模型前向传播
        Args:
            batch_dict: 输入数据字典
            return_explanations: 是否返回解释文本
            return_attention: 是否返回注意力权重
            criterion: 损失函数
        Returns:
            Dict: 预测结果和调试信息
        """
        try:
            self._validate_inputs(batch_dict)
            # 按需特征提取
            static_features, dynamic_features, text_features, extract_debug, raw_static, raw_dynamic = \
                self._extract_features(batch_dict)
            try:
                fusion_features, fusion_debug = self.temporal_alignment(
                    static_features=static_features if 'static' in self.config.feature_selection else None,
                    dynamic_features=dynamic_features if 'dynamic' in self.config.feature_selection else None,
                        text_features=text_features if 'text' in self.config.feature_selection else None,
                        person_masks=batch_dict['person_masks']
                    )
            except NaNBatchException as e:
                print(f'[跳过] 本batch因NaN/Inf被跳过: {e}')
                return None
            # 提取批次平均贡献度 (始终提取)
            static_branch_avg = fusion_debug.get('static_sub_weights_batch_avg', torch.zeros(3, device=fusion_features.device)).detach().cpu()
            dynamic_branch_avg = fusion_debug.get('dynamic_sub_weights_batch_avg', torch.zeros(2, device=fusion_features.device)).detach().cpu()
            modality_branch_avg = fusion_debug.get('modality_weights_batch_avg', torch.zeros(3, device=fusion_features.device)).detach().cpu()

            # 后续流程保持不变
            person_valid = torch.any(batch_dict['person_masks'], dim=1)
            modality_contrib = fusion_debug.get('gate_weights', None)
            static_contrib = fusion_debug.get('static_sub_weights', None)
            dynamic_contrib = fusion_debug.get('dynamic_sub_weights', None)
            
            if modality_contrib is not None:
                # 保留张量形式，移至 CPU
                modality_contrib = modality_contrib.detach().cpu()
            else:
                modality_contrib = None # 如果不存在，设为 None

            if static_contrib is not None:
                # 保留张量形式，移至 CPU
                static_contrib = static_contrib.detach().cpu()
            else:
                static_contrib = None
                
            if dynamic_contrib is not None:
                # 保留张量形式，移至 CPU
                dynamic_contrib = dynamic_contrib.detach().cpu()
            else:
                dynamic_contrib = None

            encoded = F.normalize(self.encoder(fusion_features), p=2, dim=-1)
            logits = self.classifier(encoded).squeeze(-1)
            temperature = getattr(self.config, 'temperature', 1.0)
            logits = logits / temperature
            masked_logits = logits.masked_fill(~person_valid, float('-inf'))
            probabilities = F.softmax(masked_logits, dim=1)
            pred_indices = probabilities.argmax(dim=1)
            pred_probs = probabilities.max(dim=1)[0]
            explanations_result = None
            if not getattr(self.config, 'is_ablation', False) and return_explanations:
                explanations_result = self.generate_explanation(
                    batch_dict['context_descriptions'],
                    batch_dict['person_descriptions'],
                    pred_indices,
                    batch_dict['original_ids'],
                    self.config,
                    raw_static_features=raw_static,
                    raw_dynamic_features=raw_dynamic,
                    video_ids=batch_dict.get('video_ids'),
                    target_texts=batch_dict.get('vip_explanations'),
                    target_indices=batch_dict.get('target_indices'),
                    max_length=self.config.explanation_fusion['max_length'],
                    use_template_only=getattr(self.config, 'use_template_only', False)
                )
            losses = None
            if criterion is not None:
                # 构造pairwise排序对：重要人物logit与所有非重要人物logit
                ranking_score1 = []
                ranking_score2 = []
                ranking_target = []
                if logits is not None and 'target_indices' in batch_dict:
                    for i in range(logits.shape[0]):  # 遍历batch中每个样本
                        pos_idx = batch_dict['target_indices'][i]  # 当前样本的目标人物索引
                        pos_logit = logits[i, pos_idx]             # 目标人物的logit分数
                        for neg_idx in range(logits.shape[1]):     # 遍历该样本的所有人物
                            if neg_idx != pos_idx and person_valid[i, neg_idx]:
                                neg_logit = logits[i, neg_idx]     # 有效非目标人物的logit分数
                                ranking_score1.append(pos_logit)
                                ranking_score2.append(neg_logit)
                                ranking_target.append(1)  # 重要人物应高于非重要人物
                if ranking_score1:
                    ranking_score1 = torch.stack(ranking_score1) if isinstance(ranking_score1[0], torch.Tensor) else torch.tensor(ranking_score1, device=logits.device)
                    ranking_score2 = torch.stack(ranking_score2) if isinstance(ranking_score2[0], torch.Tensor) else torch.tensor(ranking_score2, device=logits.device)
                    ranking_target = torch.tensor(ranking_target, device=logits.device, dtype=torch.float)
                    ranking_scores = (ranking_score1, ranking_score2, ranking_target)
                else:
                    ranking_scores = (torch.tensor([], device=logits.device), torch.tensor([], device=logits.device), torch.tensor([], device=logits.device))
                # 兼容 explanations_result['explanations'] 为字符串列表或字典列表
                if explanations_result and 'explanations' in explanations_result and len(explanations_result['explanations']) > 0:
                    if isinstance(explanations_result['explanations'][0], dict):
                        explanations = [p['text'] for p in explanations_result['explanations']]
                    else:
                        explanations = explanations_result['explanations']
                else:
                    explanations = None
                model_outputs = {
                    'logits': logits,
                    'features': encoded,
                    'explanations': explanations,
                    'ranking_scores': ranking_scores,
                }
                targets = {
                    'indices': batch_dict['target_indices'],
                    'explanations': batch_dict.get('vip_explanations'),
                    'valid_mask': person_valid
                }
                losses = criterion(model_outputs, targets, self)
            results = {
                'pred_indices': pred_indices,
                'probabilities': probabilities,
                'pred_probs': pred_probs,
                'logits': logits,
                'losses': losses,
                # debug_info 根据配置决定是否保存详细信息
                'debug_info': {
                    'extract': extract_debug,
                    'fusion': fusion_debug,
                } if (self.debug and getattr(self.config, 'save_debug_info', True)) else None
            }

            # 将贡献度字段添加到 results 中
            results['static_contrib'] = static_contrib
            results['dynamic_contrib'] = dynamic_contrib
            results['modality_contrib'] = modality_contrib

            if not getattr(self.config, 'is_ablation', False) and return_explanations and explanations_result:
                results.update({
                    'pred_explanations': explanations_result['explanations'],
                    'explanation_batch_loss': explanations_result.get('batch_loss', 0.0),
                    'explanation_batch_acc': explanations_result.get('batch_acc', 0.0),
                    'explanation_mean_losses': explanations_result.get('mean_losses', {}),
                    'explanation_info': explanations_result.get('explanation_info', {}),
                    'explanation_result': explanations_result # 解释生成器的原始输出
                })
            # 消融实验时，未启用分支的输出字段全部补默认安全值 # 这些是针对旧的 contrib 字段，可以根据需要删除或调整
            if getattr(self.config, 'is_ablation', False):
                if 'text' not in self.config.feature_selection:
                    results['pred_explanations'] = []

            # 确保解释相关字段始终存在（即使为空或默认值），避免下游报错
            results['pred_explanations'] = results.get('pred_explanations', [])
            results['explanation_batch_loss'] = results.get('explanation_batch_loss', 0.0)
            results['explanation_batch_acc'] = results.get('explanation_batch_acc', 0.0)
            results['explanation_mean_losses'] = results.get('explanation_mean_losses', {})
            results['explanation_info'] = results.get('explanation_info', {})
            results['explanation_result'] = results.get('explanation_result', {})

            # DDP兼容：强制所有参数都被访问一次，防止孤儿参数报错 # 已删除冗余部分
            dummy = 0.0
            _ = dummy  # 防止未使用警告

            # 将批次平均贡献度添加到 results 中 (始终添加)
            results['static_branch_avg'] = static_branch_avg
            results['dynamic_branch_avg'] = dynamic_branch_avg
            results['modality_branch_avg'] = modality_branch_avg



            return results
        except Exception as e:
            import traceback
            logger.error(f"模型前向传播失败: {str(e)}\n{traceback.format_exc()}")
            return {
                'pred_indices': None,
                'probabilities': None,
                'pred_probs': None,
                'logits': None,
                'losses': None,
                'debug_info': {'error': str(e)},
                'error': str(e)
            }

    def clear_epoch_cache(self):
        """清理epoch级别的缓存数据，释放内存"""
        if self.explanation_generator is not None and hasattr(self.explanation_generator, 'clear_epoch_cache'):
            self.explanation_generator.clear_epoch_cache()

