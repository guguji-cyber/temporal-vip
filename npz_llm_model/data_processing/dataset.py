"""
数据集定义模块
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class VIPDataset(Dataset):
    """视频重要人物检测数据集"""

    def __init__(self, config, split='train'):
        """
        初始化数据集
        
        参数:
            config: 配置对象
            split: 数据集划分，可选'train', 'val', 'test'
        """
        self.config = config
        self.split = split
        self.npz_dir = os.path.join(config.npz_dir, split)
        self.json_dir = os.path.join(config.json_dir, split)
        
        # 获取数据文件列表
        self.npz_files = self._get_npz_files()
        
        # 加载所有JSON数据
        self.json_data = {}
        for npz_file in self.npz_files:
            video_id = os.path.splitext(npz_file)[0]
            json_file = os.path.join(self.json_dir, f"{video_id}.json")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.json_data[video_id] = json.load(f)
            except Exception as e:
                print(f"加载JSON文件失败 {json_file}: {str(e)}")
                self.json_data[video_id] = {}
        
        # 测试模式下只使用部分数据（随机采样，比例由config.test_data_ratio控制）
        if config.test_mode:
            num_files = len(self.npz_files)
            ratio = getattr(config, 'test_data_ratio', 1/50)
            reduced_size = max(1, int(num_files * ratio))
            indices = np.random.choice(num_files, reduced_size, replace=False)
            self.npz_files = [self.npz_files[i] for i in sorted(indices)]
            # 同步json_data
            video_ids = [os.path.splitext(f)[0] for f in self.npz_files]
            self.json_data = {k: v for k, v in self.json_data.items() if k in video_ids}
            print(f"⚠️ 测试模式：{split}集从{num_files}随机采样到{len(self.npz_files)} (比例={ratio})")
    
    def _get_npz_files(self):
        """获取NPZ文件列表"""
        # 确保目录存在
        if not os.path.exists(self.npz_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.npz_dir}")
        
        # 获取所有NPZ文件
        npz_files = [f for f in os.listdir(self.npz_dir) if f.endswith('.npz')]
        
        # 检查每个NPZ文件是否有对应的JSON文件
        valid_files = []
        for npz_file in npz_files:
            video_id = os.path.splitext(npz_file)[0]
            json_file = os.path.join(self.json_dir, f"{video_id}.json")
            
            if os.path.exists(json_file):
                valid_files.append(npz_file)
        
        return valid_files
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        参数:
            idx: 样本索引
            
        返回:
            sample: 样本字典
        """
        # 获取视频ID
        video_id = os.path.splitext(self.npz_files[idx])[0]
        
        # 加载npz文件
        try:
            data = np.load(os.path.join(self.npz_dir, self.npz_files[idx]), allow_pickle=True)
            
            # 跳过target_index为-1的样本
            if 'target_index' in data and data['target_index'] == -1:
                print(f"[数据集] 跳过target_index为-1的样本: {video_id}")
                return self._get_empty_item()
            
            # 获取JSON数据
            json_data = self.json_data.get(video_id, {})
            
            # 从JSON数据中提取所需信息
            context_description = json_data.get('context_description', '')
            
            # 获取person_descriptions，确保长度与max_persons一致
            person_descriptions = []
            if 'person_descriptions' in json_data:
                for person in json_data['person_descriptions']:
                    desc = person.get('feature', {})
                    person_id = person.get('person_id', '')
                    # 合并所有特征描述
                    full_desc = {
                        'person_id': person_id,
                        'feature': {
                            'location': desc.get('location', ''),
                            'action': desc.get('action', ''),
                            'expression': desc.get('expression', ''),
                            'interaction': desc.get('interaction', '')
                        }
                    }
                    person_descriptions.append(full_desc)
            
            # 填充或截断person_descriptions到max_persons长度
            max_persons = self.config.max_persons
            if len(person_descriptions) < max_persons:
                person_descriptions.extend([{'person_id': '', 'feature': {}} for _ in range(max_persons - len(person_descriptions))])
            else:
                person_descriptions = person_descriptions[:max_persons]
            
            # 获取VIP解释
            # vip_explanation = json_data.get('vip_description', {}).get('explanation', '')
            vip_explanation = json_data.get('vip_description', {}).get('unconstrained_explanation', '')
            
            # 构建样本字典
            sample = {
                'video_id': video_id,
                'frames': torch.from_numpy(data['frames']).float(),
                'bboxes': torch.from_numpy(data['bboxes']).float(),
                'person_ids': torch.from_numpy(data['person_ids']).long(),
                'frame_mask': torch.from_numpy(data['frame_mask']).bool(),
                'person_mask': torch.from_numpy(data['person_mask']).bool(),
                'target_index': torch.tensor(data['target_index']).long(),
                'original_ids': torch.from_numpy(data['original_ids']).long(),
                'scene_category': str(data['scene_category']),
                'context_description': context_description,
                'person_descriptions': person_descriptions,
                'vip_explanation': vip_explanation,
                'json_data': json_data
            }
            
            return sample
            
        except Exception as e:
            print(f"加载样本失败 {video_id}: {str(e)}")
            # 返回空样本
            return self._get_empty_item()
    
    def _get_empty_item(self):
        """创建空数据项，用于处理错误情况"""
        max_frames = self.config.max_frames
        max_persons = self.config.max_persons
        frame_height = self.config.frame_height
        frame_width = self.config.frame_width
        
        data_dict = {
            'video_id': 'empty',
            'frames': torch.zeros(max_frames, frame_height, frame_width, 3).float(),
            'bboxes': torch.zeros(max_frames, max_persons, 4).float(),
            'person_ids': torch.zeros(max_frames, max_persons).long(),
            'frame_mask': torch.zeros(max_frames).bool(),
            'person_mask': torch.zeros(max_frames, max_persons).bool(),
            'target_index': torch.tensor(0).long(),
            'original_ids': torch.zeros(max_persons).long(),
            'scene_category': '',
            'context_description': '',
            'person_descriptions': [{'person_id': '', 'feature': {}} for _ in range(max_persons)],
            'vip_explanation': '',
            'json_data': {}  # 空json内容
        }
        
        return data_dict 

def my_collate_fn(batch):
    print(f"[DEBUG][collate_fn] batch size={len(batch)}")
    for i, item in enumerate(batch):
        for k, v in item.items():
            print(f"  batch[{i}][{k}]: type={type(v)}, shape={getattr(v, 'shape', None)}, device={getattr(v, 'device', None)}")