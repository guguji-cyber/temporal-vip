"""
训练管理器模块
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
import logging
from datetime import datetime
import traceback
import csv
import torch.nn.functional as F
import torch.distributed as dist
import gc

# 恢复相对导入
from ..utils.logger import Logger
from ..utils.metrics import Metrics, normalize_metrics
from ..utils.visualization import Visualizer
from ..train.losses import TextSimilarityLoss, TotalLoss

class Trainer:
    """训练管理器类"""
    
    def __init__(self, model, data_loader, config):
        """
        初始化训练管理器
        
        参数:
            model: 模型对象
            data_loader: 数据加载器对象
            config: 配置对象
        """
        self.config = config
        self.data_loader = data_loader
        
        # 从配置中获取对比损失权重,默认为0.3
        self.contrastive_loss_weight = getattr(config, 'contrastive_loss_weight', 0.3)
        
        # 初始化日志记录器
        self.logger = Logger(config)
        
        # 创建invalid samples日志文件路径
        self.invalid_samples_path = os.path.join(config.log_dir, 'invalid_samples.log')
        
        # 加载已知的无效样本
        self.known_invalid_samples = set()
        if os.path.exists(self.invalid_samples_path):
            with open(self.invalid_samples_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'Video IDs:' in line:
                        video_ids = eval(line.split('Video IDs:')[1].strip())
                        if isinstance(video_ids, (list, tuple)):
                            self.known_invalid_samples.update(video_ids)
        
        # 初始化每轮的无效样本统计
        self.invalid_samples_count = 0
        
        # 初始化CUDA环境
        self._init_cuda_environment()
        
        # 将模型移动到设备
        self.model = model.to(self.device)
        
        # 多GPU训练设置
        self.is_distributed = config.multi_gpu and torch.cuda.device_count() > 1
        if self.is_distributed:
            self.setup_distributed()
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        
        # 获取所有需要优化的参数
        model_to_use = self.model.module if self.is_distributed else self.model
        base_lr = self.config.learning_rate

        # 确保参数不重复
        backbone_params = []
        fusion_params = []
        text_params = []
        time_params = []

        # 收集backbone参数
        for name, param in model_to_use.named_parameters():
            if 'fusion' in name:
                fusion_params.append(param)
            elif 'text' in name:
                text_params.append(param)
            elif 'time_weights' in name:
                time_params.append(param)
            else:
                backbone_params.append(param)

        # 构建参数组
        parameters = [
            {
                'params': backbone_params,
                'lr': base_lr
            },
            {
                'params': fusion_params,
                'lr': base_lr * 5
            },
            {
                'params': text_params,
                'lr': base_lr * 2
            }
        ]
        
        # 如果存在time_weights且不为空，添加到参数组
        if time_params:
            parameters.append({
                'params': time_params,
                'lr': base_lr
            })
        
        # 优化器
        self.optimizer = optim.Adam(
            parameters,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            T_mult=2,
            eta_min=base_lr * 0.01
        )
        
        # 损失函数
        self.criterion = TotalLoss(
            cls_weight=getattr(config, 'cls_loss_weight', 1.0),
            text_weight=getattr(config, 'text_loss_weight', 0.5),
            contrastive_weight=getattr(config, 'contrastive_loss_weight', 0.3),
            reg_weight=getattr(config, 'reg_weight', 0.0001),
            config=config
        )
        
        # 度量指标
        self.metrics = Metrics()
        
        # 设置实验名称
        # 目录全部由config统一管理
        
        # 初始化日志记录器
        
        # 可视化工具
        self.visualizer = Visualizer(config)
        
        # 训练状态
        self.current_epoch = 0
        self.best_rank1_accuracy = 0.0
        self.best_loss = float('inf')
        
        # 新增属性
        self.compute_text_loss = getattr(config, 'compute_text_loss', True)
        
        # 初始化混合精度训练
        self.scaler = GradScaler('cuda')
        
        # 设置CUDA内存分配器
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        # 设置模型保存路径
        self.model_save_dir = config.checkpoint_dir
        # 目录已在config中创建，无需重复创建
        
        # 新增权重历史属性
        self.static_weights_history = []
        self.dynamic_weights_history = []
        self.fusion_weights_history = []
        
        self.invalid_samples_dir = self.config.log_dir
        # 目录已在config中创建，无需重复创建
        self.all_invalid_video_ids = set()
    
    def _init_cuda_environment(self):
        """初始化CUDA环境"""
        try:
            # 从环境变量获取local_rank
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # 设置设备
            if torch.cuda.is_available():
                if self.config.multi_gpu:
                    self.device = torch.device(f'cuda:{self.local_rank}')
                else:
                    self.device = torch.device('cuda:0')
                    
                # 确保当前进程使用正确的GPU
                torch.cuda.set_device(self.device)
                
                # 设置CUDA内存分配器
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                
                # 设置CUDA错误处理
                torch.cuda.set_device(self.device)
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                self.logger.info(f"使用GPU设备: {device_name} (index: {current_device})")
                
                # 检查GPU内存
                total_memory = torch.cuda.get_device_properties(current_device).total_memory
                allocated_memory = torch.cuda.memory_allocated(current_device)
                self.logger.info(f"GPU总内存: {total_memory/1024**3:.2f}GB")
                self.logger.info(f"已分配内存: {allocated_memory/1024**3:.2f}GB")
            else:
                self.device = torch.device('cpu')
                self.logger.info("CUDA不可用，使用CPU进行训练")
                
        except Exception as e:
            error_msg = f"CUDA环境初始化失败: {str(e)}\n{traceback.format_exc()}"
            self.logger.info(error_msg)
            raise RuntimeError(error_msg)
    
    def setup_distributed(self):
        """设置分布式训练环境"""
        try:
            if dist.is_initialized():
                self.logger.info("分布式环境已初始化")
                return
                
            # 获取本地排名和世界大小
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            
            # 初始化分布式环境
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
            
            # 更新设备
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.logger.info(f"分布式训练初始化成功: rank={self.local_rank}, world_size={self.world_size}")
            
        except Exception as e:
            error_msg = f"分布式训练初始化失败: {str(e)}\n{traceback.format_exc()}"
            self.logger.info(error_msg)
            raise RuntimeError(error_msg)
    
    def _move_to_device(self, batch_dict):
        """
        将批次数据移动到指定设备，并加断言和详细日志
        """
        try:
            if batch_dict is None:
                return None
            # 数据完整性断言与日志
            def assert_batch_integrity(batch_dict):
                ti = batch_dict.get('target_indices', None)
                oi = batch_dict.get('original_ids', None)
                pm = batch_dict.get('person_masks', None)
                if ti is not None:
                    assert isinstance(ti, torch.Tensor), f"target_indices不是Tensor: {type(ti)}"
                    assert ti.dtype == torch.long, f"target_indices类型错误: {ti.dtype}"
                    assert (ti >= 0).all(), f"target_indices存在负数: {ti}"
                    N = oi.shape[-1] if oi is not None else 20
                    assert (ti < N).all(), f"target_indices越界: {ti}, N={N}"
                if oi is not None:
                    assert isinstance(oi, torch.Tensor), f"original_ids不是Tensor: {type(oi)}"
                    assert oi.dtype == torch.long, f"original_ids类型错误: {oi.dtype}"
                    assert (oi >= 0).all(), f"original_ids存在负数: {oi}"
                if pm is not None:
                    assert isinstance(pm, torch.Tensor), f"person_masks不是Tensor: {type(pm)}"
                    assert pm.dtype in [torch.bool, torch.uint8, torch.int64], f"person_masks类型错误: {pm.dtype}"
            try:
                assert_batch_integrity(batch_dict)
            except Exception as e:
                print(f"[断言失败] batch_dict关键字段异常: {e}")
                print(f"  target_indices: {batch_dict.get('target_indices', None)}")
                print(f"  original_ids: {batch_dict.get('original_ids', None)}")
                print(f"  person_masks: {batch_dict.get('person_masks', None)}")
                return None
            device_batch = {}
            for key, value in batch_dict.items():
                try:
                    if isinstance(value, torch.Tensor):
                        device_batch[key] = value.to(self.device, non_blocking=True)
                    elif isinstance(value, (list, tuple)):
                        device_batch[key] = value
                    else:
                        device_batch[key] = value
                except Exception as e:
                    tqdm.write(f"移动数据到设备失败 (key={key}): {str(e)}")
                    continue
            return device_batch
        except Exception as e:
            tqdm.write(f"批次数据移动到设备失败: {str(e)}")
            return None

    def save_checkpoint(self, epoch, metrics, extra_info=None):
        """
        专业保存模型检查点，支持多模块，保存完整训练状态
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self._serialize_config(self.config),
            'extra_info': extra_info or {}
        }
        # 只在非消融实验时保存每个epoch模型
        # if not getattr(self.config, 'is_ablation', False):
        #     main_save_path = os.path.join(self.config.main_model_dir, f'epoch_{epoch+1}.pth')
        #     torch.save(checkpoint, main_save_path)
        #     self.logger.info(f'主模型已保存至: {main_save_path}')
        #     # 解释生成器单独保存
        #     model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        #     if hasattr(model_to_use, 'explanation_generator') and not getattr(self.config, 'is_ablation', False):
        #         eg = model_to_use.explanation_generator
        #         eg_ckpt = {'epoch': epoch, 'state_dict': eg.state_dict()}
        #         eg_save_path = os.path.join(self.config.explanation_generator_dir, f'epoch_{epoch+1}.pth')
        #         torch.save(eg_ckpt, eg_save_path)
        #         self.logger.info(f'解释生成器权重保存至: {eg_save_path}')
        # 最佳模型分别保存
        is_main_process = (not self.is_distributed) or (dist.get_rank() == 0)
        if is_main_process and metrics.get('rank1_accuracy', 0) > getattr(self, 'best_rank1_acc', 0):
            self.best_rank1_acc = metrics['rank1_accuracy']
            main_best_path = os.path.join(self.config.main_model_dir, 'best.pth')
            torch.save(checkpoint, main_best_path)
            self.logger.info(f'主模型最佳权重保存至: {main_best_path} (rank1_accuracy={metrics["rank1_accuracy"]:.4f})')
            # model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
            # if hasattr(model_to_use, 'explanation_generator') and not getattr(self.config, 'is_ablation', False):
            #     eg = model_to_use.explanation_generator
            #     eg_best_path = os.path.join(self.config.explanation_generator_dir, 'best.pth')
            #     torch.save({'epoch': epoch, 'state_dict': eg.state_dict()}, eg_best_path)
            #     self.logger.info(f'解释生成器最佳权重保存至: {eg_best_path}')

    def save_records_readme(self):
        """生成records目录下的README.txt说明文件"""
        readme_path = os.path.join(self.config.records_dir, 'README.txt')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(
                'main_model/train_metrics.csv: ID模型训练集每轮损失与准确率\n'
                'main_model/val_metrics.csv: ID模型验证集每轮损失与准确率\n'
                'explanation_generator/train_metrics.csv: 文本生成模型训练集损失与准确率\n'
                'explanation_generator/val_metrics.csv: 文本生成模型验证集损失与准确率\n'
            )

    def train_epoch(self, epoch):
        """
        训练一个epoch，返回平均损失和准确率
        """
        self.model.train()
        train_metrics = {
            'total_loss': 0.0,
            'cls_loss': 0.0,
            'text_loss': 0.0,
            'contrastive_loss': 0.0,
            'reg_loss': 0.0,
            'text_similarity': 0.0,
            'correct_count': 0,
            'total_count': 0,
            'rank1_accuracy': 0.0,
            'ranking_loss': 0.0,
            'total_pairs': 0,
        }
        processed_batches = 0
        eg_total_loss = 0.0
        eg_bart_loss = 0.0
        eg_semantic_loss = 0.0
        eg_feature_loss = 0.0
        eg_batches = 0
        # 初始化列表，用于收集每个批次的分支权重
        all_static_branch_avg = []
        all_dynamic_branch_avg = []
        all_modality_branch_avg = []
        train_loader = self.data_loader.get_train_loader()
        is_main_process = not self.is_distributed or dist.get_rank() == 0
        train_iter = tqdm(train_loader, desc=f"Training {epoch+1}/{self.config.epochs}", dynamic_ncols=True, disable=not is_main_process)
        try:
            for batch_idx, batch_data in enumerate(train_iter):
                target_indices = batch_data.get('target_indices', None)
                if target_indices is not None and (target_indices == -1).any():
                    invalid_indices = (target_indices == -1).nonzero(as_tuple=True)[0].tolist()
                    original_ids = batch_data.get('original_ids', None)
                    video_ids = batch_data.get('video_ids', None)
                    if video_ids and all(vid in self.known_invalid_samples for vid in video_ids):
                        continue
                    if is_main_process and video_ids:
                        new_invalid_videos = [vid for vid in video_ids if vid not in self.known_invalid_samples]
                        if new_invalid_videos:
                            with open(self.invalid_samples_path, 'a', encoding='utf-8') as f:
                                f.write(f"Batch {batch_idx}:\n")
                                f.write(f"  Invalid indices: {invalid_indices}\n")
                                if original_ids is not None:
                                    f.write(f"  Original IDs: {original_ids.tolist()}\n")
                                f.write(f"  Video IDs: {new_invalid_videos}\n")
                                f.write("-" * 50 + "\n")
                            self.known_invalid_samples.update(new_invalid_videos)
                            self.all_invalid_video_ids.update(new_invalid_videos)
                batch_data = self._move_to_device(batch_data)
                self.optimizer.zero_grad()

                # 定期清理内存
                if batch_idx % getattr(self.config, 'memory_cleanup_interval', 100) == 0:
                    self.config.cleanup_gpu_memory()

                try:
                    with autocast('cuda'):
                        outputs = self.model(batch_data, criterion=self.criterion)
                        if outputs is None or not isinstance(outputs, dict):
                            tqdm.write(f"批次{batch_idx} outputs类型异常，已跳过。outputs={outputs}")
                            continue
                        losses = outputs.get('losses', None)
                        pred_indices = outputs.get('pred_indices', None)
                        target_indices = batch_data.get('target_indices', None)
                        if losses is None:
                            tqdm.write(f"批次{batch_idx} 损失为None，已跳过。outputs={outputs}")
                            continue
                        batch_size = pred_indices.size(0) if pred_indices is not None else 0
                        # 修正：loss统计方式与验证一致，累加mean loss × batch_size
                        train_metrics['total_loss'] += float(losses.get('total_loss', 0.0)) * batch_size
                        train_metrics['cls_loss'] += float(losses.get('classification_loss', 0.0)) * batch_size
                        train_metrics['text_loss'] += float(losses.get('text_similarity_loss', 0.0)) * batch_size
                        train_metrics['contrastive_loss'] += float(losses.get('contrastive_loss', 0.0)) * batch_size
                        train_metrics['reg_loss'] += float(losses.get('regularization_loss', 0.0)) * batch_size
                        # text_similarity统计
                        similarities = losses.get('text_similarities', None)
                        if similarities is not None:
                            if isinstance(similarities, torch.Tensor):
                                train_metrics['text_similarity'] += similarities.sum().item()
                            elif isinstance(similarities, (list, np.ndarray)):
                                train_metrics['text_similarity'] += float(np.sum(similarities))
                        # 准确率统计
                        if pred_indices is not None and target_indices is not None:
                            correct = (pred_indices == target_indices).sum().item()
                            train_metrics['correct_count'] += correct
                            train_metrics['total_count'] += batch_size
                        eg_mean_losses = outputs.get('explanation_mean_losses', None)
                        if eg_mean_losses is not None:
                            eg_total_loss += eg_mean_losses.get('total', 0.0)
                            eg_bart_loss += eg_mean_losses.get('bart', 0.0)
                            eg_semantic_loss += eg_mean_losses.get('semantic', 0.0)
                            eg_feature_loss += eg_mean_losses.get('feature', 0.0)
                            eg_batches += 1
                        loss = losses['total_loss'] if ('total_loss' in losses) else None
                    if loss is not None:
                        self.scaler.scale(loss).backward()
                        if self.config.clip_grad_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config.clip_grad_norm
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    # 权重收集
                    if outputs is not None:
                        if 'static_branch_avg' in outputs and outputs['static_branch_avg'] is not None:
                            val = outputs['static_branch_avg']
                            if isinstance(val, torch.Tensor):
                                val = val.detach().cpu().numpy()
                            all_static_branch_avg.append(val)
                        if 'dynamic_branch_avg' in outputs and outputs['dynamic_branch_avg'] is not None:
                            val = outputs['dynamic_branch_avg']
                            if isinstance(val, torch.Tensor):
                                val = val.detach().cpu().numpy()
                            all_dynamic_branch_avg.append(val)
                        if 'modality_branch_avg' in outputs and outputs['modality_branch_avg'] is not None:
                            val = outputs['modality_branch_avg']
                            if isinstance(val, torch.Tensor):
                                val = val.detach().cpu().numpy()
                            all_modality_branch_avg.append(val)
                    processed_batches += 1
                    # 进度条显示当前累计均值
                    avg_loss = train_metrics['total_loss'] / train_metrics['total_count'] if train_metrics['total_count'] > 0 else 0.0
                    avg_acc = train_metrics['correct_count'] / train_metrics['total_count'] if train_metrics['total_count'] > 0 else 0.0
                    # 统计pair数
                    num_pairs = 0
                    if 'ranking_scores' in losses and losses['ranking_scores'] is not None:
                        score1, score2, ranking_target = losses['ranking_scores']
                        num_pairs = int(score1.numel())
                    train_metrics['ranking_loss'] += float(losses.get('ranking_loss', 0.0)) * num_pairs
                    train_metrics['total_pairs'] += num_pairs
                    train_metrics['rank1_accuracy'] = avg_acc
                    train_iter.set_postfix({
                        'a': f"{avg_acc:.4f}",
                        'l': f"{avg_loss:.4f}",
                        'cls_l': f"{train_metrics['cls_loss'] / train_metrics['total_count'] if train_metrics['total_count'] > 0 else 0.0:.4f}",
                        'rank_l': f"{train_metrics['ranking_loss'] / train_metrics['total_pairs'] if train_metrics['total_pairs'] > 0 else 0.0:.4f}",
                        # 'sem_l': f"{eg_semantic_loss / eg_batches if eg_batches > 0 else 0.0:.4f}",
                        'txt_l': f"{train_metrics['text_loss'] / train_metrics['total_count'] if train_metrics['total_count'] > 0 else 0.0:.4f}",
                        'ctra_l': f"{train_metrics['contrastive_loss'] / train_metrics['total_count'] if train_metrics['total_count'] > 0 else 0.0:.4f}",
                        'reg_l': f"{train_metrics['reg_loss'] / train_metrics['total_count'] if train_metrics['total_count'] > 0 else 0.0:.4f}",
                    })
                except Exception as e:
                    import traceback
                    tqdm.write(f"批次处理失败: {str(e)}\n{traceback.format_exc()}")
                    tqdm.write(f"[异常batch数据] batch_idx={batch_idx}, batch_data keys={list(batch_data.keys()) if batch_data else None}")
                    continue
            # 权重收集
            if processed_batches > 0:
                if all_static_branch_avg:
                    train_metrics['static_branch_avg'] = (np.mean(all_static_branch_avg, axis=0)).tolist()
                if all_dynamic_branch_avg:
                    train_metrics['dynamic_branch_avg'] = (np.mean(all_dynamic_branch_avg, axis=0)).tolist()
                if all_modality_branch_avg:
                    train_metrics['modality_branch_avg'] = (np.mean(all_modality_branch_avg, axis=0)).tolist()

            # 清理权重收集列表，释放内存
            del all_static_branch_avg, all_dynamic_branch_avg, all_modality_branch_avg
            gc.collect()
            # 分布式指标all_reduce汇总
            if self.is_distributed:
                metrics_tensor = torch.tensor([
                    train_metrics['total_loss'],
                    train_metrics['cls_loss'],
                    train_metrics['text_loss'],
                    train_metrics['contrastive_loss'],
                    train_metrics['reg_loss'],
                    train_metrics['text_similarity'],
                    train_metrics['correct_count'],
                    train_metrics['total_count'],
                    train_metrics['ranking_loss'],
                    train_metrics['total_pairs']
                ], dtype=torch.float64, device='cuda')
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                # 汇总后重新赋值
                total_count = metrics_tensor[7].item()
                train_metrics['total_loss'] = metrics_tensor[0].item()
                train_metrics['cls_loss'] = metrics_tensor[1].item()
                train_metrics['text_loss'] = metrics_tensor[2].item()
                train_metrics['contrastive_loss'] = metrics_tensor[3].item()
                train_metrics['reg_loss'] = metrics_tensor[4].item()
                train_metrics['text_similarity'] = metrics_tensor[5].item()
                train_metrics['correct_count'] = metrics_tensor[6].item()
                train_metrics['total_count'] = total_count
                train_metrics['rank1_accuracy'] = metrics_tensor[6].item() / total_count if total_count > 0 else 0.0
                train_metrics['ranking_loss'] = metrics_tensor[8].item()
                train_metrics['total_pairs'] = int(metrics_tensor[9].item())
            # 统一归一化（无论单卡还是分布式）
            train_metrics = normalize_metrics(
                train_metrics,
                count_key='total_count',
                exclude_keys=['total_pairs', 'static_branch_avg', 'dynamic_branch_avg', 'modality_branch_avg', 'correct_count', 'rank1_accuracy', 'total_count']
            )
            train_metrics['rank1_accuracy'] = train_metrics['correct_count'] / train_metrics['total_count'] if train_metrics['total_count'] > 0 else 0.0
            # 统一交由visualizer收集、写入csv
            if is_main_process:
                self.visualizer.record_epoch_metrics(epoch, train_metrics, mode='train', is_main_process=True)

            # epoch结束后清理缓存数据
            model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_to_use, 'clear_epoch_cache'):
                model_to_use.clear_epoch_cache()

            # 强制清理GPU缓存
            self.config.cleanup_gpu_memory()

            return train_metrics
        except Exception as e:
            tqdm.write(f"训练epoch {epoch} 失败: {str(e)}")
            return {k: 0.0 for k in train_metrics}

    def train(self, evaluator):
        """
        训练模型
        """
        # 训练开始前同步所有进程
        if self.is_distributed:
            dist.barrier()

        is_main_process = not self.is_distributed or dist.get_rank() == 0

        # 训练开始前清理内存
        self.config.cleanup_gpu_memory()
        if is_main_process:
            print(f"[Training Start] {self.config.get_memory_info()}")
        if is_main_process:
            self.logger.log(f"开始训练，总计{self.config.epochs}轮次...")
            print(f"开始训练，总计{self.config.epochs}轮次...")
        # 新增：打印实际采样到的训练/验证集样本数
        train_loader = self.data_loader.get_train_loader()
        val_loader = self.data_loader.get_val_loader()
        if is_main_process:
            print(f"⚠️ 实际训练样本数: {len(train_loader.dataset)}")
            print(f"⚠️ 实际验证样本数: {len(val_loader.dataset)}")
        if self.config.test_mode:
            if is_main_process:
                print(f"⚠️ 测试模式已启用：只使用{self.config.test_data_ratio}比例的数据进行训练和评估")
        # 早停机制参数
        early_stop_patience = getattr(self.config, 'early_stop_patience', None)
        patience_counter = 0
        best_rank1_accuracy = self.best_rank1_accuracy if hasattr(self, 'best_rank1_accuracy') else 0.0
        best_epoch = self.current_epoch
        # 训练循环
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            # 训练一个轮次
            start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            train_time = time.time() - start_time
            # 打印训练指标
            is_main_process = not self.is_distributed or self.local_rank == 0
            if is_main_process:
                tqdm.write(f"Epoch {epoch+1}/{self.config.epochs} - " + 
                     f"Train Loss: {train_metrics['total_loss']:.4f}, " +
                     f"Train Acc: {train_metrics['rank1_accuracy']:.4f}, " +
                    #  f"Rank Loss: {train_metrics['ranking_loss'] / train_metrics['total_pairs'] if train_metrics['total_pairs'] > 0 else 0.0:.4f}, " +
                     f"Time: {train_time:.2f}s")
                # 记录所有训练指标到csv
                self.visualizer.record_epoch_metrics(epoch, train_metrics, mode='train', is_main_process=True)
            # 每隔eval_interval轮进行评估
            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = evaluator.evaluate(epoch=epoch, save_results=True)
                if is_main_process:
                    # 记录所有验证指标到csv
                    self.visualizer.record_epoch_metrics(epoch, val_metrics, mode='val', is_main_process=True)
                    tqdm.write(f"Epoch {epoch+1}/{self.config.epochs} - " + 
                        f"Val Loss: {val_metrics['total_loss']:.4f}, " + 
                        f"Val Acc: {val_metrics['rank1_accuracy']:.4f}, " +
                        # f"Rank Loss: {val_metrics['ranking_loss'] / val_metrics['total_pairs'] if val_metrics['total_pairs'] > 0 else 0.0:.4f}, " +
                        f"Text Sim: {val_metrics['text_similarity']:.4f}")
                # 保存前同步
                if self.is_distributed:
                    dist.barrier()

                # 保存当前epoch模型和最优模型
                self.save_checkpoint(epoch, val_metrics)

            # 每个epoch结束后清理内存
            model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_to_use, 'clear_epoch_cache'):
                model_to_use.clear_epoch_cache()

            self.config.cleanup_gpu_memory()

            if is_main_process:
                print(f"[Epoch {epoch+1}] {self.config.get_memory_info()}")
                # 早停机制：监控验证集准确率
                current_acc = val_metrics.get('rank1_accuracy', 0.0)
                if current_acc > best_rank1_accuracy:
                    best_rank1_accuracy = current_acc
                    best_epoch = epoch
                    patience_counter = 0
                    # 兼容分布式/单卡
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    best_path = os.path.join(self.config.main_model_dir, 'best.pth')
                    torch.save(model_to_save.state_dict(), best_path)
                    if is_main_process:
                        self.logger.info(f'最佳模型已保存至: {best_path} (rank1_accuracy={best_rank1_accuracy:.4f})')
                else:
                    patience_counter += 1
                    if is_main_process and early_stop_patience is not None:
                        print(f"[EarlyStopping] 验证集准确率连续未提升: {patience_counter}/{early_stop_patience}")
                if patience_counter >= (early_stop_patience if early_stop_patience is not None else 1e9):
                    if is_main_process:
                        print(f"[EarlyStopping] 触发早停: epoch={epoch}, best_epoch={best_epoch}, best_acc={best_rank1_accuracy:.4f}")
                        self.logger.info(f"[EarlyStopping] 触发早停: epoch={epoch}, best_epoch={best_epoch}, best_acc={best_rank1_accuracy:.4f}")
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    return
        if is_main_process:
            print("训练完成！")
            self.logger.log("训练完成！")
        if hasattr(self.logger, 'get_log_summary'):
            summary = self.logger.get_log_summary()
            if is_main_process:
                print(summary)
        # 训练完成或早停后，销毁分布式进程组，防止NCCL资源泄漏
        if dist.is_initialized():
            dist.destroy_process_group()

    def load_checkpoint(self, checkpoint_path, strict=False, load_optimizer=True, load_scheduler=True):
        """
        专业加载模型检查点，支持严格/兼容模式，详细log所有不匹配参数
        """
        is_main_process = not self.is_distributed or dist.get_rank() == 0
        if is_main_process:
            print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint['model_state_dict']
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        # 兼容分布式/单卡
        try:
            missing, unexpected = model_to_use.load_state_dict(model_state, strict=strict)
            if not strict:
                if is_main_process:
                    print(f"[兼容模式] 未加载参数: {missing}")
                    print(f"[兼容模式] 多余参数: {unexpected}")
        except Exception as e:
            if is_main_process:
                print(f"模型权重加载异常: {e}")
        # 优化器/调度器
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                if is_main_process:
                    print(f'优化器参数加载失败: {e}')
        if load_scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                if is_main_process:
                    print(f'调度器参数加载失败: {e}')
        self.current_epoch = checkpoint.get('epoch', 0)
        if is_main_process:
            print(f'已加载模型，当前epoch: {self.current_epoch}')
        return self.current_epoch

    @staticmethod
    def _serialize_config(config):
        """智能序列化config，兼容各种类型"""
        if hasattr(config, 'to_dict'):
            return config.to_dict()
        elif hasattr(config, '__dict__'):
            return dict(config.__dict__)
        else:
            return str(config)
