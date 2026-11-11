"""
文本编码器模块
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    """文本编码器"""
    
    def __init__(self, config):
        """
        初始化文本编码器
        
        参数:
            config: 配置对象
        """
        super(TextEncoder, self).__init__()
        self.config = config
        
        # 使用BERT模型作为编码器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        
        # 冻结BERT参数（可选）
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        # 特征维度
        self.bert_dim = 768
        # self.hidden_dim = config.hidden_dim
        self.hidden_dim = config.text_extractor['feature_dim']
        
        # 特征投影层
        self.projector = nn.Linear(self.bert_dim, self.hidden_dim)
    
    def encode(self, text_list, max_length=128):
        """
        编码文本列表
        
        参数:
            text_list: 文本列表
            max_length: 最大序列长度
        
        返回:
            embeddings: 文本嵌入向量
        """
        # 处理空文本
        text_list = [text if text else "[PAD]" for text in text_list]
        
        # 对文本进行分词
        encoded_dict = self.tokenizer(
            text_list,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 将编码移至设备
        input_ids = encoded_dict['input_ids'].to(self.bert_model.device)
        attention_mask = encoded_dict['attention_mask'].to(self.bert_model.device)
        
        # 使用BERT提取特征
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS]标记的嵌入作为整个序列的表示
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 投影到目标维度
        projected_embeddings = self.projector(embeddings)
        
        return projected_embeddings
    
    def forward(self, text_list, max_length=128):
        """
        前向传播
        
        参数:
            text_list: 文本列表
            max_length: 最大序列长度
        
        返回:
            embeddings: 投影后的文本嵌入向量
        """
        return self.encode(text_list, max_length) 