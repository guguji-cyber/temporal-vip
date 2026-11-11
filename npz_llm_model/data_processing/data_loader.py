"""
数据加载器模块
"""
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os

from .dataset import VIPDataset


def collate_fn(batch):
    """
    自定义批次整合函数
    
    参数:
        batch: 数据批次，列表形式，每个元素包含固定大小的数据:
        - frames: (120, 192, 336, 3)
        - bboxes: (120, 20, 4)
        - person_ids: (120, 20)
        - frame_mask: (120,)
        - person_mask: (120, 20)
        - target_index: ()
        - original_ids: (20,)
    
    返回:
        batch_dict: 整合后的批次字典
    """
    # 筛选有效数据项（防止空数据项）
    valid_batch = [item for item in batch if item['video_id'] != 'empty']
    
    # 如果所有数据都无效，返回空批次
    if len(valid_batch) == 0:
        return None
    
    # 断言与日志
    for i, item in enumerate(valid_batch):
        ti = item.get('target_index', None)
        oi = item.get('original_ids', None)
        if ti is not None:
            assert isinstance(ti, torch.Tensor), f"[collate_fn] target_index不是Tensor: {type(ti)}"
            assert ti.dtype == torch.long, f"[collate_fn] target_index类型错误: {ti.dtype}"
            assert ti.item() >= 0, f"[collate_fn] target_index存在负数: {ti}"
            N = oi.shape[-1] if oi is not None else 20
            assert ti.item() < N, f"[collate_fn] target_index越界: {ti}, N={N}"
        if oi is not None:
            assert isinstance(oi, torch.Tensor), f"[collate_fn] original_ids不是Tensor: {type(oi)}"
            assert oi.dtype == torch.long, f"[collate_fn] original_ids类型错误: {oi.dtype}"
            assert (oi >= 0).all(), f"[collate_fn] original_ids存在负数: {oi}"
    
    # 初始化批次字典
    batch_dict = {
        'video_ids': [],
        'frames': [],
        'bboxes': [],
        'person_ids': [],
        'frame_masks': [],
        'person_masks': [],
        'target_indices': [],
        'original_ids': [],
        'scene_categories': [],
        'context_descriptions': [],
        'person_descriptions': [],
        'vip_explanations': [],
        'json_datas': []
    }
    
    # 填充批次字典
    for item in valid_batch:
        batch_dict['video_ids'].append(item['video_id'])
        batch_dict['frames'].append(item['frames'])
        batch_dict['bboxes'].append(item['bboxes'])
        batch_dict['person_ids'].append(item['person_ids'])
        batch_dict['frame_masks'].append(item['frame_mask'])
        batch_dict['person_masks'].append(item['person_mask'])
        batch_dict['target_indices'].append(item['target_index'])
        batch_dict['original_ids'].append(item['original_ids'])
        batch_dict['scene_categories'].append(item['scene_category'])
        batch_dict['context_descriptions'].append(item['context_description'])
        batch_dict['person_descriptions'].append(item['person_descriptions'])
        batch_dict['vip_explanations'].append(item['vip_explanation'])
        batch_dict['json_datas'].append(item.get('json_data', {}))
    
    # 转换为张量形式
    batch_dict['frames'] = torch.stack(batch_dict['frames'])  # [B, 120, 192, 336, 3]
    batch_dict['bboxes'] = torch.stack(batch_dict['bboxes'])  # [B, 120, 20, 4]
    batch_dict['person_ids'] = torch.stack(batch_dict['person_ids'])  # [B, 120, 20]
    batch_dict['frame_masks'] = torch.stack(batch_dict['frame_masks'])  # [B, 120]
    batch_dict['person_masks'] = torch.stack(batch_dict['person_masks'])  # [B, 120, 20]
    batch_dict['target_indices'] = torch.stack(batch_dict['target_indices'])  # [B]
    batch_dict['original_ids'] = torch.stack(batch_dict['original_ids'])  # [B, 20]
    
    return batch_dict


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, config):
        """
        初始化数据加载器
        
        参数:
            config: 配置对象
        """
        self.config = config
        
        # 从环境变量获取local_rank
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # 设置设备
        if torch.cuda.is_available():
            if config.multi_gpu:
                self.device = torch.device(f'cuda:{self.local_rank}')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.is_distributed = config.multi_gpu and dist.is_initialized()
        
        # 创建数据集
        self.train_dataset = VIPDataset(config, split='train')
        self.val_dataset = VIPDataset(config, split='val')
        self.test_dataset = VIPDataset(config, split='test')
        
        # 创建数据采样器（分布式训练时使用）
        train_sampler = DistributedSampler(self.train_dataset) if self.is_distributed else None
        val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.is_distributed else None
        test_sampler = DistributedSampler(self.test_dataset, shuffle=False) if self.is_distributed else None
        
        # 根据训练模式优化配置
        is_distributed = getattr(config, 'multi_gpu', False)

        if is_distributed:
            # DDP模式：保守配置
            num_workers = min(config.num_workers if hasattr(config, 'num_workers') else 2, 2)
            pin_memory = False
            persistent_workers = False
            prefetch_factor = 2
        else:
            # 单GPU模式：可以更激进
            num_workers = min(config.num_workers if hasattr(config, 'num_workers') else 4, os.cpu_count() - 1)
            pin_memory = True
            persistent_workers = num_workers > 0
            prefetch_factor = 4

        # 如果没有 worker，prefetch_factor 必须为 None
        if num_workers == 0:
            prefetch_factor = None
        
        # 创建数据加载器
        self.train_loader = TorchDataLoader(
            dataset=self.train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            collate_fn=self._optimized_collate_fn,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        
        self.val_loader = TorchDataLoader(
            dataset=self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._optimized_collate_fn,
            sampler=val_sampler,
            drop_last=False,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        
        self.test_loader = TorchDataLoader(
            dataset=self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._optimized_collate_fn,
            sampler=test_sampler,
            drop_last=False,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
    
    def _optimized_collate_fn(self, batch):
        """
        优化的批次整合函数
        """
        try:
            # 筛选有效数据项
            valid_batch = [item for item in batch if item['video_id'] != 'empty']
            if len(valid_batch) == 0:
                return None
                
            # 断言与日志
            for i, item in enumerate(valid_batch):
                ti = item.get('target_index', None)
                oi = item.get('original_ids', None)
                if ti is not None:
                    assert isinstance(ti, torch.Tensor), f"[_optimized_collate_fn] target_index不是Tensor: {type(ti)}"
                    assert ti.dtype == torch.long, f"[_optimized_collate_fn] target_index类型错误: {ti.dtype}"
                    assert ti.item() >= 0, f"[_optimized_collate_fn] target_index存在负数: {ti}"
                    N = oi.shape[-1] if oi is not None else 20
                    assert ti.item() < N, f"[_optimized_collate_fn] target_index越界: {ti}, N={N}"
                if oi is not None:
                    assert isinstance(oi, torch.Tensor), f"[_optimized_collate_fn] original_ids不是Tensor: {type(oi)}"
                    assert oi.dtype == torch.long, f"[_optimized_collate_fn] original_ids类型错误: {oi.dtype}"
                    assert (oi >= 0).all(), f"[_optimized_collate_fn] original_ids存在负数: {oi}"
            
            # 初始化批次字典
            batch_dict = {
                'video_ids': [],
                'frames': [],
                'bboxes': [],
                'person_ids': [],
                'frame_masks': [],
                'person_masks': [],
                'target_indices': [],
                'original_ids': [],
                'scene_categories': [],
                'context_descriptions': [],
                'person_descriptions': [],
                'vip_explanations': [],
                'json_datas': []
            }
            
            # 填充批次字典
            for item in valid_batch:
                try:
                    # 张量数据
                    batch_dict['frames'].append(item['frames'])
                    batch_dict['bboxes'].append(item['bboxes'])
                    batch_dict['person_ids'].append(item['person_ids'])
                    batch_dict['frame_masks'].append(item['frame_mask'])
                    batch_dict['person_masks'].append(item['person_mask'])
                    batch_dict['target_indices'].append(item['target_index'])
                    batch_dict['original_ids'].append(item['original_ids'])
                    
                    # 非张量数据
                    batch_dict['video_ids'].append(item['video_id'])
                    batch_dict['scene_categories'].append(item['scene_category'])
                    batch_dict['context_descriptions'].append(item['context_description'])
                    batch_dict['person_descriptions'].append(item['person_descriptions'])
                    batch_dict['vip_explanations'].append(item['vip_explanation'])
                    batch_dict['json_datas'].append(item.get('json_data', {}))
                except Exception as e:
                    print(f"处理数据项时出错: {str(e)}")
                    continue
            
            # 转换为张量形式
            try:
                batch_dict['frames'] = torch.stack(batch_dict['frames'])
                batch_dict['bboxes'] = torch.stack(batch_dict['bboxes'])
                batch_dict['person_ids'] = torch.stack(batch_dict['person_ids'])
                batch_dict['frame_masks'] = torch.stack(batch_dict['frame_masks'])
                batch_dict['person_masks'] = torch.stack(batch_dict['person_masks'])
                batch_dict['target_indices'] = torch.stack(batch_dict['target_indices'])
                batch_dict['original_ids'] = torch.stack(batch_dict['original_ids'])
            except Exception as e:
                print(f"张量转换失败: {str(e)}")
                return None
            
            return batch_dict
            
        except Exception as e:
            print(f"批次整合失败: {str(e)}")
            return None
    
    def get_train_loader(self):
        """获取训练数据加载器"""
        return self.train_loader
    
    def get_val_loader(self):
        """获取验证数据加载器"""
        return self.val_loader
    
    def get_test_loader(self):
        """获取测试数据加载器"""
        return self.test_loader
    
    def get_train_dataset(self):
        """获取训练数据集"""
        return self.train_dataset
    
    def get_val_dataset(self):
        """获取验证数据集"""
        return self.val_dataset
    
    def get_test_dataset(self):
        """获取测试数据集"""
        return self.test_dataset