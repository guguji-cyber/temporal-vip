"""
文本特征提取模块 - 使用BERT处理文本特征
整合场景描述和人物描述的特征提取
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class TextProjector(nn.Module):
    """文本特征投影器"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.text_extractor['bert_feature_dim'], 
                     config.text_extractor['projector_hidden']),
            nn.LayerNorm(config.text_extractor['projector_hidden']),
            nn.Dropout(config.text_extractor['dropout']),
            nn.ReLU(),
            nn.Linear(config.text_extractor['projector_hidden'], 
                     config.text_extractor['feature_dim']),
            nn.LayerNorm(config.text_extractor['feature_dim']),
            nn.Dropout(config.text_extractor['dropout'])
        )
    
    def forward(self, x):
        return self.net(x)

class TextFeatureExtractor(nn.Module):
    """文本特征提取器"""
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        
        # BERT初始化
        try:
            self.tokenizer = BertTokenizer.from_pretrained(config.text_extractor['bert_path'])
            self.model = BertModel.from_pretrained(config.text_extractor['bert_path']).to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.error(f"BERT初始化失败: {str(e)}")
            raise RuntimeError("BERT初始化失败")
            
        # 特征维度
        self.bert_dim = config.text_extractor['bert_feature_dim']
        self.feature_dim = config.text_extractor['feature_dim']
        
        # 特征投影
        self.scene_projector = TextProjector(config)
        self.person_projector = TextProjector(config)
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=config.text_extractor['attention_heads'],
            dropout=config.text_extractor['dropout'],
            batch_first=True
        )
        
        # 特征门控
        self.feature_gate = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(config.text_extractor['dropout']),
            nn.Sigmoid()
        )
        
        self.register_buffer("empty_features", torch.zeros(1, self.feature_dim))
        
    @torch.amp.autocast('cuda')
    def encode_text(self, text_list: List[str]) -> torch.Tensor:
        """文本编码"""
        try:
            text_list = [str(text) if text else "[PAD]" for text in text_list]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    text_list,
                    max_length=self.config.text_extractor['max_length'],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = F.normalize(embeddings, dim=1)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"文本编码失败: {str(e)}")
            return torch.zeros(len(text_list), self.bert_dim, device=self.device)
            
    def process_person_description(self, description: Optional[Union[Dict[str, Any], str]]) -> str:
        """处理人物描述"""
        try:
            if isinstance(description, str):
                return description
                
            if isinstance(description, dict) and 'feature' in description:
                feature = description['feature']
                text_parts = []
                
                for field in self.config.text_extractor['feature_fields']:
                    if field in feature and feature[field]:
                        text_parts.append(str(feature[field]))
                
                return "，".join(text_parts) if text_parts else ""
                
            return ""
            
        except Exception as e:
            logger.warning(f"处理人物描述时出错: {str(e)}")
            return ""
            
    @torch.amp.autocast('cuda')
    def enhance_scene_features(
        self,
        scene_features: torch.Tensor,
        person_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """特征增强"""
        try:
            batch_size, num_persons, _ = person_features.shape
            expanded_scene = scene_features.unsqueeze(1).expand(-1, num_persons, -1)
            
            # 交叉注意力
            attn_output, attn_weights = self.cross_attention(
                query=person_features,
                key=expanded_scene,
                value=expanded_scene
            )
            
            # 特征门控
            gate = self.feature_gate(torch.cat([person_features, attn_output], dim=-1))
            gate = gate * self.config.text_extractor['gate_ratio']
            
            # 特征融合
            enhanced_features = gate * person_features + (1 - gate) * attn_output
            
            debug_info = {
                "attention_weights": attn_weights.detach(),
                "gate_values": gate.detach(),
                "scene_contribution": ((1 - gate) * attn_output).detach(),
                "person_contribution": (gate * person_features).detach()
            }
            
            return enhanced_features, debug_info
            
        except Exception as e:
            logger.error(f"特征增强失败: {str(e)}")
            return torch.zeros_like(person_features), {"error": str(e)}
            
    @torch.amp.autocast('cuda')
    def forward(
        self,
        context_descriptions: List[str],
        person_descriptions: List[List[Dict[str, Any]]],
        original_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """前向传播"""
        try:
            batch_size = len(context_descriptions)
            num_persons = original_ids.shape[1]
            debug_info = {}
            
            # 场景特征提取
            scene_features = self.encode_text(context_descriptions)
            scene_features = self.scene_projector(scene_features)
            debug_info["scene_features"] = scene_features.detach()
            
            # 人物特征提取
            person_text_features = torch.zeros(
                batch_size, num_persons, self.feature_dim,
                device=self.device
            )
            
            all_desc = []
            desc_indices = []
            for b in range(batch_size):
                id_to_index = {
                    str(id_.item()): idx 
                    for idx, id_ in enumerate(original_ids[b])
                    if id_.item() != 0
                }
                
                for desc in person_descriptions[b]:
                    person_id = str(desc.get('person_id', ''))
                    if person_id in id_to_index:
                        idx = id_to_index[person_id]
                        processed_desc = self.process_person_description(desc)
                        if processed_desc:
                            all_desc.append(processed_desc)
                            desc_indices.append((b, idx))
            
            if all_desc:
                encoded_features = self.encode_text(all_desc)  # [num_valid, bert_dim]
                projected_features = self.person_projector(encoded_features)  # [num_valid, feature_dim]
                for i, (b, idx) in enumerate(desc_indices):
                    person_text_features[b, idx] = projected_features[i]
            
            debug_info["person_features"] = person_text_features.detach()
            
            # 特征增强
            enhanced_features, enhance_info = self.enhance_scene_features(
                scene_features, person_text_features
            )
            debug_info.update(enhance_info)
            
            return enhanced_features, debug_info
            
        except Exception as e:
            logger.error(f"文本特征提取失败: {str(e)}")
            return (
                self.empty_features.expand(batch_size, num_persons, -1),
                {"error": str(e)}
            ) 