"""
解释生成器模块 - 多目标优化（语言模型 + 特征利用 + 语义相关）
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BertTokenizer
)
from glob import glob
import logging
import json
from ..train.losses import TextSimilarityLoss
from torch.nn import CrossEntropyLoss
import math
from sentence_transformers import SentenceTransformer
import re
from ..models.temporal_alignment import AttentionPooling
import numpy as np
import gc

# 配置专门的解释生成器日志
explain_logger = logging.getLogger("explain_generator")
explain_logger.setLevel(logging.INFO)

def safe_index(tensor, *indices):
    assert tensor is not None, "tensor为None"
    assert tensor.dim() >= len(indices), f"tensor shape异常: {tensor.shape}, indices={indices}"
    return tensor[indices]

class TemplateGenerator(nn.Module):
    """优化后的模板生成器"""
    def __init__(self, config):
        super().__init__()
        # 自动适配静态/动态特征名称
        self.static_names = config.static_extractor['feature_names']
        self.dynamic_names = config.dynamic_extractor['feature_names']
        self.feature_names = self.static_names + self.dynamic_names  # 总特征顺序
        # 优化分支描述模板，只保留突出档
        self.feature_desc_map = self._init_feature_desc_map()
        # 自动适配特征重要性学习层
        self.feature_importance = nn.Linear(len(self.feature_names), len(self.feature_names))
        self.feature_importance.weight.data.normal_(mean=0.0, std=0.02)
        if self.feature_importance.bias is not None:
            self.feature_importance.bias.data.zero_()
        # 设置设备
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        self.feature_importance = self.feature_importance.to(self.device)
        # 特征阈值和权重（可选，自动适配长度）
        self.score_thresholds = {name: 0.5 for name in self.feature_names}
        self.feature_weights = {name: 1.0 for name in self.feature_names}

    def get_feature_scores(self, raw_static_dict, raw_dynamic_dict, batch_idx, person_idx):
        """计算特征得分，自动适配特征类型，输入已为聚合后特征"""
        device = next(self.feature_importance.parameters()).device
        scores = {}
        # AttentionPooling实例缓存，避免重复创建
        if not hasattr(self, '_attn_poolers'):
            self._attn_poolers = {}
        # 静态特征
        for name in self.static_names:
            if name in raw_static_dict:
                feat = raw_static_dict[name][batch_idx, person_idx]  # [T, D] or [D] or 标量
                if feat.dim() == 2:  # [T, D]
                    if name not in self._attn_poolers:
                        self._attn_poolers[name] = AttentionPooling(feat.shape[-1]).to(feat.device)
                    pooled = self._attn_poolers[name](feat.unsqueeze(0)).squeeze(0)  # [D]
                    val = pooled.mean()
                elif feat.dim() > 0:
                    val = feat.mean()
                else:
                    val = feat
                scores[name] = torch.sigmoid(val if torch.isfinite(val) else torch.tensor(0.0, device=device)).item()
            else:
                scores[name] = 0.0
        # 动态特征
        for name in self.dynamic_names:
            if name in raw_dynamic_dict:
                feat = raw_dynamic_dict[name][batch_idx, person_idx]
                if feat.dim() == 2:  # [T, D]
                    if name not in self._attn_poolers:
                        self._attn_poolers[name] = AttentionPooling(feat.shape[-1]).to(feat.device)
                    pooled = self._attn_poolers[name](feat.unsqueeze(0)).squeeze(0)
                    val = pooled.mean()
                elif feat.dim() > 0:
                    val = feat.mean()
                else:
                    val = feat
                scores[name] = torch.sigmoid(val if torch.isfinite(val) else torch.tensor(0.0, device=device)).item()
            else:
                scores[name] = 0.0
        importance_input = torch.tensor([scores[n] for n in self.feature_names], device=device)
        importance = torch.sigmoid(self.feature_importance(importance_input))
        return scores, importance
    
    def get_scene_feature_scores(self, raw_static_dict, raw_dynamic_dict, batch_idx, valid_indices):
        """只对有效人物索引列表计算分支得分，返回: {分支: [有效人数分数]}"""
        device = next(self.feature_importance.parameters()).device
        scores_dict = {name: [] for name in self.feature_names}
        for person_idx in valid_indices:
            scores, _ = self.get_feature_scores(raw_static_dict, raw_dynamic_dict, batch_idx, person_idx)
            for name in self.feature_names:
                scores_dict[name].append(scores[name])
        return scores_dict

    def get_feature_desc(self, scores, scene_scores=None, valid_person_idx=None, threshold=0.7):
        """只描述分位高于阈值的分支，且只用正向突出描述"""
        desc = []
        for name in self.feature_names:
            val = scores.get(name, 0.0)
            if scene_scores is not None and valid_person_idx is not None:
                arr = np.array(scene_scores[name])
                sort_idx = np.argsort(-arr)
                rank = np.where(sort_idx == valid_person_idx)[0][0]
                percentile = 1.0 - rank / max(len(arr)-1, 1)
                if percentile >= threshold:
                    desc.append(self.feature_desc_map[name])
            # else: 不描述
        return desc
        
    # def _init_feature_desc_map(self):
    #     return {
    #         'area': "画面占比突出",
    #         'centrality': "处于视觉中央位置",
    #         'clarity': "其面部表情丰富",
    #         'action': "肢体动作幅度明显",
    #         'speech': "其发言频率较高"
    #     }
        
    def _init_feature_desc_map(self):
        """返回英文名词短语，保证单条/多条都能直接嵌入句子"""
        return {
            'area':       "pronounced visual footprint",
            'centrality': "central spatial position",
            'clarity':    "rich and discernible facial expressions",
            'action':     "markedly expansive bodily movements",
            'optical_flow': "pronounced motion patterns (optical flow)",
            'speech':     "high frequency of verbal contributions"
        }
        
    def forward(self, raw_static_features, raw_dynamic_features, context_desc, person_desc, batch_idx, person_idx, valid_indices=None, threshold=0.7):
        # 支持有效人物集合分位筛选
        if valid_indices is not None and len(valid_indices) > 1:
            scene_scores = self.get_scene_feature_scores(raw_static_features, raw_dynamic_features, batch_idx, valid_indices)
            valid_person_idx = valid_indices.index(person_idx)
        else:
            scene_scores = None
            valid_person_idx = None
        scores, importance = self.get_feature_scores(raw_static_features, raw_dynamic_features, batch_idx, person_idx)
        feature_descs = self.get_feature_desc(scores, scene_scores, valid_person_idx, threshold=threshold)
        
        # if not feature_descs:
        #     return "该人物在本场景与其他人有一定的交互关系，因此被认为是重要人物。"# return "The individual is deemed salient due to their interactive relationships with others within the scene."
        # main_desc = "，".join(feature_descs)
        # final_desc = f"该人物相较于场景中的其他人而言，{main_desc}，因此被认为是本场景中的重要人物。"
        # return final_desc

        # 没有任何显著静态特征时
        if not feature_descs:
            return "The individual is deemed salient due to their interactive relationships with others within the scene."

        # 拼接多条：1 条无逗号；≥2 条用逗号+and
        if len(feature_descs) == 1:
            main_desc = feature_descs[0]
        else:
            main_desc = ', '.join(feature_descs[:-1]) + ' and ' + feature_descs[-1]

        final_desc = (f"Relative to other individuals in the scene, the person exhibits "
                    f"{main_desc} and is consequently identified as a key figure.")
        return final_desc

class ExplanationGenerator(nn.Module):
    """解释生成器 - 模板生成和BART生成串联"""
    def __init__(self, config):
        super().__init__()
        # 1. 模板生成器
        self.template_generator = TemplateGenerator(config)
        
        try:
            # 2. 加载BART模型
            bart_path = config.bart_path
            
            # 加载分词器和模型
            self.tokenizer = BertTokenizer.from_pretrained(bart_path)
            self.model = BartForConditionalGeneration.from_pretrained(bart_path)
            
            # 加载文本相似度模型
            self.text_encoder = SentenceTransformer(config.st_path)
            
            # 3. 设置设备
            self.device = config.device if hasattr(config, 'device') else 'cpu'
            self.model = self.model.to(self.device)
            
            # 损失权重参数（log_sigma更稳定）
            self.log_sigma_bart = nn.Parameter(torch.zeros(1))
            self.log_sigma_semantic = nn.Parameter(torch.zeros(1))
            self.log_sigma_feature = nn.Parameter(torch.zeros(1))
            

            
        except Exception as e:
            raise RuntimeError(f'模型加载失败: {str(e)}')

    def clear_epoch_cache(self):
        """清理epoch级别的缓存数据，释放内存"""
        # 这里可以添加需要在epoch结束时清理的数据
        # 目前解释生成器没有跨batch的累积数据，所以暂时只做垃圾回收
        gc.collect()


            
    def prepare_input_text(self, template_text, context_desc, person_desc):
        """准备BART模型的输入文本"""
        return f"""场景：{context_desc}
输入：{template_text}
输出："""

    def generate_explanation(self, input_text):
        """使用BART模型生成解释文本"""
        # 编码输入
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            return_token_type_ids=False  # 明确不返回token_type_ids
        ).to(self.device)
        
        # 生成文本
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=150,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
    def compute_feature_usage_score(self, generated_text, feature_dict, static_scores, dynamic_scores):
        """计算特征利用度分数
        Args:
            generated_text: str, 生成的解释文本
            feature_dict: dict, 人物描述特征
            static_scores: dict, 静态特征分数
            dynamic_scores: dict, 动态特征分数
        Returns:
            float: 特征利用度分数
        """
        score = 0.0
        total_features = 0
        
        # 1. 检查静态特征的使用
        static_keywords = {
            'centrality': ['中心', '边缘', '画面'],
            'area': ['占比', '体型', '大小'],
            'clarity': ['清晰', '面部', '脸部']
        }
        
        for feature, keywords in static_keywords.items():
            if static_scores.get(feature, 0) > 0.5:  # 如果特征显著
                total_features += 1
                if any(kw in generated_text for kw in keywords):
                    score += 1.0
                    
        # 2. 检查动态特征的使用
        dynamic_keywords = {
            'action': ['动作', '行为', '移动'],
            'speech': ['说话', '发言', '交谈']
        }
        
        for feature, keywords in dynamic_keywords.items():
            if dynamic_scores.get(feature, 0) > 0.5:  # 如果特征显著
                total_features += 1
                if any(kw in generated_text for kw in keywords):
                    score += 1.0
                    
        # 3. 检查文本描述特征的使用
        if isinstance(feature_dict, dict) and 'feature' in feature_dict:
            for feat_type, desc in feature_dict['feature'].items():
                if desc:  # 如果有描述
                    total_features += 1
                    # 检查描述的关键信息是否在生成文本中
                    key_phrases = desc.split('，')  # 按逗号分割描述
                    for phrase in key_phrases:
                        if phrase in generated_text:
                            score += 1.0
                            break
        
        return score / max(total_features, 1)  # 避免除零
        
    def compute_semantic_relevance(self, generated_text, target_text, feature_dict):
        """计算语义相关性分数
        Args:
            generated_text: str, 生成的解释文本
            target_text: str, 目标解释文本
            feature_dict: dict, 特征信息
        Returns:
            float: 语义相关性分数
        """
        # 1. 计算与目标文本的语义相似度
        gen_embedding = self.text_encoder.encode(generated_text, convert_to_tensor=True)
        tgt_embedding = self.text_encoder.encode(target_text, convert_to_tensor=True)
        similarity = F.cosine_similarity(gen_embedding.unsqueeze(0), tgt_embedding.unsqueeze(0))
        similarity = (similarity.item() + 1) / 2  # 归一化到[0,1]
        # 2. 计算与特征描述的语义相关性
        feature_text = ""
        if isinstance(feature_dict, dict) and 'feature' in feature_dict:
            feature_text = "。".join(f"{k}:{v}" for k, v in feature_dict['feature'].items() if v)
        if feature_text:
            feat_embedding = self.text_encoder.encode(feature_text, convert_to_tensor=True)
            feature_similarity = F.cosine_similarity(gen_embedding.unsqueeze(0), feat_embedding.unsqueeze(0))
            feature_similarity = (feature_similarity.item() + 1) / 2  # 归一化到[0,1]
            # 综合两种相似度
            return (similarity + feature_similarity) / 2
        return similarity

    def forward(self, context_descriptions, person_descriptions, pred_indices, original_ids, config,
                raw_static_features, raw_dynamic_features, video_ids, target_texts=None, target_indices=None, max_length=200, use_template_only=False):
        results = {
            'explanations': [],
            'losses': {
                'bart': [],      # BART基础损失
                'semantic': [],   # 语义相关性损失
                'feature': [],    # 特征利用度损失
                'total': []      # 总损失
            },
            'video_info': [],
            'explanation_info': None
        }
        batch_size = len(context_descriptions)
        for batch_idx in range(batch_size):
            context_desc = context_descriptions[batch_idx]
            video_id = video_ids[batch_idx]
            if self.training and target_indices is not None:
                curr_indices = [target_indices[batch_idx].item() if isinstance(target_indices[batch_idx], torch.Tensor) 
                              else target_indices[batch_idx]]
            else:
                if pred_indices.dim() == 0:
                    curr_indices = [pred_indices.item()]
                elif pred_indices.dim() == 1:
                    curr_indices = [pred_indices[batch_idx].item()]
                else:
                    curr_indices = pred_indices[batch_idx].tolist()
                if not isinstance(curr_indices, list):
                    curr_indices = [curr_indices]
            for person_idx in curr_indices:
                if person_idx >= len(original_ids[batch_idx]) or original_ids[batch_idx][person_idx] == 0:
                    continue
                if person_idx >= len(person_descriptions[batch_idx]):
                    continue
                person_desc = person_descriptions[batch_idx][person_idx]
                results['video_info'].append({
                    'video_id': video_id,
                    'batch_idx': batch_idx,
                    'person_idx': person_idx,
                    'original_id': original_ids[batch_idx][person_idx].item()
                })
                # 只用模板时不拼接person_desc
                valid_indices = [i for i, oid in enumerate(original_ids[batch_idx]) if oid != 0]
                template_text = self.template_generator(
                    raw_static_features,
                    raw_dynamic_features,
                    context_desc,
                    person_desc,
                    batch_idx,
                    person_idx,
                    valid_indices=valid_indices,
                    threshold=0.7
                )
                if use_template_only:
                    results['explanations'].append(template_text)
                    continue
                # 拼接person_desc到模板后，作为BART输入
                if isinstance(person_desc, dict) and 'feature' in person_desc:
                    person_text = []
                    for n in self.template_generator.feature_names:
                        if n.lower() in person_desc['feature']:
                            person_text.append(f"{n}：{person_desc['feature'][n.lower()]}")
                    person_desc_text = "，".join(person_text)
                    template_text = template_text + f"根据描述，{person_desc_text}。"
                input_text = self.prepare_input_text(template_text, context_desc, person_desc)
                # 训练或验证阶段都计算loss
                if (self.training and target_texts is not None) or (not self.training and target_texts is not None):
                    target_text = target_texts[batch_idx]
                    if isinstance(target_text, list):
                        target_text = target_text[0]
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding=True,
                        return_token_type_ids=False
                    ).to(self.device)
                    generated_ids = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=150,
                        num_beams=4,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        length_penalty=1.0,
                        no_repeat_ngram_size=3
                    )
                    generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    labels = self.tokenizer(
                        target_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding=True,
                        return_token_type_ids=False
                    ).input_ids.to(self.device)
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        labels=labels
                    )
                    bart_loss = outputs.loss
                    static_scores, _ = self.template_generator.get_feature_scores(
                        raw_static_features, raw_dynamic_features, batch_idx, person_idx
                    )
                    feature_usage_score = self.compute_feature_usage_score(
                        generated_text, 
                        person_desc,
                        static_scores,
                        {}  # 动态分数暂时为空
                    )
                    feature_loss = 1 - feature_usage_score
                    semantic_score = self.compute_semantic_relevance(
                        generated_text,
                        target_text,
                        person_desc
                    )
                    semantic_loss = 1 - semantic_score
                    total_loss = (
                        torch.exp(-self.log_sigma_bart) * bart_loss +
                        torch.exp(-self.log_sigma_semantic) * semantic_loss +
                        torch.exp(-self.log_sigma_feature) * feature_loss +
                        (self.log_sigma_bart + self.log_sigma_semantic + self.log_sigma_feature)
                    )
                    results['losses']['bart'].append(bart_loss.detach().cpu() if isinstance(bart_loss, torch.Tensor) else torch.tensor(bart_loss))
                    results['losses']['semantic'].append(semantic_loss.detach().cpu() if isinstance(semantic_loss, torch.Tensor) else torch.tensor(semantic_loss))
                    results['losses']['feature'].append(torch.tensor(feature_loss))
                    results['losses']['total'].append(total_loss.detach().cpu() if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss))
                    results['explanation_info'] = {
                        'target_text': target_text,
                        'generated_text': generated_text,
                        'bart_loss': float(bart_loss.item()) if hasattr(bart_loss, 'item') else float(bart_loss),
                        'semantic_loss': float(semantic_loss.item()) if hasattr(semantic_loss, 'item') else float(semantic_loss),
                        'feature_loss': float(feature_loss),
                        'total_loss': float(total_loss.item()) if hasattr(total_loss, 'item') else float(total_loss),
                        'feature_usage_score': float(feature_usage_score),
                        'semantic_score': float(semantic_score.item()) if hasattr(semantic_score, 'item') else float(semantic_score)
                    }
                else:
                    generated_text = self.generate_explanation(input_text)
                results['explanations'].append({
                    'video_id': video_id,
                    'text': generated_text
                })
        # 计算平均损失
        if results['losses']['total']:
            results['mean_losses'] = {
                'bart': torch.stack(results['losses']['bart']).mean().item(),
                'semantic': torch.stack(results['losses']['semantic']).mean().item(),
                'feature': torch.stack(results['losses']['feature']).mean().item(),
                'total': torch.stack(results['losses']['total']).mean().item()
            }

        return results