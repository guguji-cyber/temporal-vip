"""
预测器模块 - 与训练评估模块完全对齐
"""
import os
import time
import torch
import json
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.amp import autocast
import logging
import traceback
from typing import Dict, List, Optional
import csv

from ..utils.logger import Logger
from ..utils.metrics import Metrics
from ..utils.visualization import Visualizer
from ..models.enhanced_transformer_model import EnhancedTransformerModel
from ..data_processing.data_loader import DataLoader

class Predictor:
    """预测器类 - 与训练评估模块完全对齐"""
    
    def __init__(self, model: EnhancedTransformerModel, data_loader: DataLoader, config):
        """
        初始化预测器
        
        Args:
            model: 模型对象
            data_loader: 数据加载器对象
            config: 配置对象
        """
        self.config = config
        self.model = model
        self.data_loader = data_loader
        
        # 初始化日志记录器和可视化工具（与训练评估保持一致）
        self.logger = Logger(config)
        self.metrics = Metrics()
        self.visualizer = Visualizer(config)
        
        # 初始化CUDA环境
        self._init_cuda_environment()
        
        # 加载最佳模型权重
        self._load_best_model()
        
        # 将模型移动到设备
        self.model = self.model.to(self.device)
        
        # 多GPU设置（与训练评估保持一致）
        self.is_distributed = config.multi_gpu and torch.cuda.device_count() > 1
        if self.is_distributed:
            self.setup_distributed()
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
        
        # 混合精度设置
        self.use_amp = getattr(self.config, 'use_amp', False) and self.device.type == 'cuda'
        
        # 预测结果保存目录（使用config中的标准目录）
        self.pred_dir = config.pred_dir
        os.makedirs(self.pred_dir, exist_ok=True)
        
        # 性能统计
        self.performance_stats = {
            'total_samples': 0,
            'total_time': 0.0,
            'avg_time_per_sample': 0.0
        }
    
    def _init_cuda_environment(self):
        """初始化CUDA环境（与训练器保持一致）"""
        try:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if torch.cuda.is_available():
                if self.config.multi_gpu:
                    self.device = torch.device(f'cuda:{self.local_rank}')
                else:
                    self.device = torch.device('cuda:0')
                torch.cuda.set_device(self.device)
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
            else:
                self.device = torch.device('cpu')
        except Exception as e:
            self.logger.error(f"CUDA环境初始化失败: {str(e)}")
            raise
    
    def setup_distributed(self):
        """设置分布式环境（与训练器保持一致）"""
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://'
            )
    
    def _move_to_device(self, batch_dict):
        """ 移动数据到设备（与评估器保持一致）"""
        try:
            if batch_dict is None:
                return None
                
            # 数据完整性检查（与评估器保持一致）
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
            
            # 执行完整性检查
            assert_batch_integrity(batch_dict)
            
            # 移动数据到设备
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_dict.items()
            }
        except Exception as e:
            self.logger.error(f"数据移动到设备失败: {str(e)}")
            return None
    
    def _load_best_model(self):
        """加载模型权重：优先使用指定路径，默认使用当前训练的最佳模型"""
        try:
            # 1. 确定模型路径
            model_path = getattr(self.config, 'model_path', None)
            if not model_path:
                # 未指定时使用当前训练的最佳模型
                model_path = os.path.join(self.config.main_model_dir, 'best.pth')
                self.logger.info("未指定模型路径，将使用当前训练的最佳模型")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            self.logger.info(f"正在加载模型权重: {model_path}")
            
            # 2. 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 3. 处理checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # 标准checkpoint格式
                    state_dict = checkpoint['model_state_dict']
                    # 记录训练信息
                    self.best_epoch = checkpoint.get('epoch', -1)
                    self.best_metrics = checkpoint.get('metrics', {})
                    self.best_rank1_accuracy = self.best_metrics.get('rank1_accuracy', 0.0)
                    self.best_loss = self.best_metrics.get('loss', float('inf'))
                    # 记录配置信息
                    self.checkpoint_config = checkpoint.get('config', {})
                else:
                    # 简单的state_dict格式
                    state_dict = checkpoint
                    self.logger.warning("模型文件为简单state_dict格式，无额外信息")
            else:
                raise ValueError("无效的模型文件格式")
            
            # 4. 加载权重
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            
            # 5. 记录加载情况
            if missing:
                self.logger.warning(f"模型加载 - 缺失的键: {missing}")
            if unexpected:
                self.logger.warning(f"模型加载 - 意外的键: {unexpected}")
            
            # 6. 输出加载信息
            load_info = [
                f"模型加载完成:",
                f"- 路径: {model_path}",
                f"- 是否为指定模型: {'是' if getattr(self.config, 'model_path', None) else '否'}"
            ]
            
            # 如果有训练信息则添加
            if hasattr(self, 'best_metrics'):
                load_info.extend([
                    f"- 最佳轮次: {self.best_epoch}",
                    f"- 最佳Rank-1准确率: {self.best_rank1_accuracy:.4f}",
                    f"- 最佳损失: {self.best_loss:.4f}",
                    f"- 其他指标: {', '.join(f'{k}: {v:.4f}' for k, v in self.best_metrics.items() if k not in ['rank1_accuracy', 'loss'])}"
                ])
            
            self.logger.info("\n".join(load_info))
            
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            raise
    
    @torch.inference_mode()
    def predict(self, split: str = 'test', epoch: int = None) -> List[Dict]:
        """
        在指定数据集上进行预测（与评估器的评估流程保持一致）
        
        Args:
            split: 数据集划分 ('train', 'val', 'test')
            
        Returns:
            predictions: 预测结果列表
        """
        # 确保模型处于评估模式
        self.model.eval()
        torch.set_grad_enabled(False)  # 双重保险，确保不计算梯度
        
        all_predictions = []
        start_time = time.time()
        
        # 获取数据加载器
        data_loader = getattr(self.data_loader, f'get_{split}_loader')()
        if not data_loader:
            raise ValueError(f"无效的数据集划分: {split}")
        
        # 设置进度条
        is_distributed = getattr(self, 'is_distributed', False)
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1
        is_main_process = (not is_distributed) or (rank == 0)
        pbar = tqdm(data_loader, desc=f"Predicting ({split})", disable=not is_main_process)
        
        # === 新增loss/acc统计 ===
        total_loss_sum = 0.0
        total_count = 0
        correct_count = 0
        # Rank-k counters
        rank1_count = 0
        rank2_count = 0
        rank3_count = 0
        
        for batch in pbar:
            try:
                # 移动数据到设备（与评估器保持一致）
                batch = self._move_to_device(batch)
                if batch is None:
                    continue
                
                # 使用AMP进行推理
                with autocast('cuda'):
                    outputs = self.model(batch)
                
                if outputs is None:
                    continue
                
                # === 新增loss/acc统计 ===
                # Always increment total_count by processed batch size for correct denominator
                batch_size = len(batch['video_ids'])
                total_count += batch_size

                # Safely extract losses dict
                losses_dict = outputs.get('losses') if isinstance(outputs.get('losses'), dict) else None
                if losses_dict and 'total_loss' in losses_dict:
                    mean_loss = float(losses_dict['total_loss'])
                    total_loss_sum += mean_loss * batch_size

                # Robust accuracy counting: handle scalars, tensors, lists
                if 'pred_indices' in outputs and 'target_indices' in batch:
                    try:
                        pred_t = torch.as_tensor(outputs['pred_indices']).cpu().view(-1)
                        targ_t = torch.as_tensor(batch['target_indices']).cpu().view(-1)
                        min_n = min(pred_t.numel(), targ_t.numel())
                        if min_n == 0:
                            correct = 0
                        else:
                            correct = int((pred_t[:min_n] == targ_t[:min_n]).sum().item())
                    except Exception as e:
                        self.logger.error(f"计算准确率时出错，默认0: {e}")
                        correct = 0
                    correct_count += correct
                
                # Rank@k counting when probability distributions are available
                probs = outputs.get('probabilities') if outputs.get('probabilities') is not None else None
                if probs is not None and 'target_indices' in batch:
                    try:
                        probs_t = torch.as_tensor(probs).cpu()
                        topk = torch.topk(probs_t, k=3, dim=1).indices  # [B, 3]
                        targets = torch.as_tensor(batch['target_indices']).cpu().view(-1)
                        # Ensure shapes align
                        min_n = min(topk.shape[0], targets.numel())
                        if min_n > 0:
                            t = targets[:min_n]
                            tk = topk[:min_n]
                            rank1_count += int((tk[:, :1] == t.unsqueeze(1)).any(dim=1).sum().item())
                            rank2_count += int((tk[:, :2] == t.unsqueeze(1)).any(dim=1).sum().item())
                            rank3_count += int((tk[:, :3] == t.unsqueeze(1)).any(dim=1).sum().item())
                    except Exception as e:
                        self.logger.error(f"Rank@k计算失败，跳过: {e}")
                        pass
                
                # 处理预测结果
                batch_predictions = self._process_batch_predictions(batch, outputs)
                all_predictions.extend(batch_predictions)
                
                # 更新进度条
                if is_main_process:
                    pbar.set_postfix({
                        'batch_size': len(batch['video_ids']),
                        'total': len(all_predictions),
                        'loss': f"{(total_loss_sum / total_count) if total_count > 0 else '-'}",
                        'acc': f"{(correct_count / total_count) if total_count > 0 else '-'}"
                    })
            
            except Exception as e:
                # 记录失败的批次信息（包含video_ids）以便复现与调试
                try:
                    failed_ids = batch.get('video_ids') if isinstance(batch, dict) and 'video_ids' in batch else None
                except Exception:
                    failed_ids = None
                error_msg = f"预测批次出错: {str(e)}\n{traceback.format_exc()}\nFailed video_ids: {failed_ids}"
                self.logger.error(error_msg)
                # 追加写入失败日志文件
                try:
                    fail_log_path = os.path.join(self.pred_dir, 'failed_batches.txt')
                    with open(fail_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().isoformat()}\tFailed IDs: {failed_ids}\tError: {str(e)}\n")
                except Exception as ex:
                    self.logger.error(f"写入失败日志失败: {ex}")
                continue
        
        # 分布式收集所有进程的预测结果和loss/acc
        if is_distributed:
            gathered_predictions = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_predictions, all_predictions)
            if rank == 0:
                all_predictions = sum(gathered_predictions, [])
                for pred in all_predictions:
                    for k, v in pred.items():
                        if isinstance(v, torch.Tensor):
                            pred[k] = v.cpu()
            else:
                all_predictions = []
            # === 新增loss/acc同步 ===
            tensor = torch.tensor([total_loss_sum, total_count, correct_count, rank1_count, rank2_count, rank3_count], dtype=torch.float64, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_loss_sum, total_count, correct_count, rank1_count, rank2_count, rank3_count = tensor.tolist()
        
        # 统计性能
        total_time = time.time() - start_time
        self.performance_stats.update({
            'total_samples': len(all_predictions),
            'total_time': total_time,
            'avg_time_per_sample': total_time / len(all_predictions) if all_predictions else 0
        })
        
        # 保存预测结果（使用标准目录）
        if is_main_process:
            self._save_predictions(all_predictions, split, epoch=epoch)
            self.logger.info(
                f"预测完成 - 总样本数: {self.performance_stats['total_samples']}, "
                f"总耗时: {self.performance_stats['total_time']:.2f}s, "
                f"平均每样本: {self.performance_stats['avg_time_per_sample']*1000:.2f}ms"
            )
            # === 新增loss/acc输出 ===
            if total_count > 0:
                rank1 = rank1_count / total_count if total_count > 0 else 0.0
                rank2 = rank2_count / total_count if total_count > 0 else 0.0
                rank3 = rank3_count / total_count if total_count > 0 else 0.0
                print(f"Test Loss: {total_loss_sum / total_count:.4f}, Test Acc: {correct_count / total_count:.4f}")
                print(f"Rank-1: {rank1:.4f}, Rank-2: {rank2:.4f}, Rank-3: {rank3:.4f}")
        return all_predictions
    
    def _process_batch_predictions(self, batch: Dict, outputs: Dict) -> List[Dict]:
         """处理批次预测结果（格式与评估器保持一致）"""
         batch_predictions = []
         losses = outputs.get('losses') if isinstance(outputs.get('losses'), dict) else None
         pred_indices = outputs.get('pred_indices') if outputs.get('pred_indices') is not None else None
         probs = outputs.get('probabilities') if outputs.get('probabilities') is not None else None
         
         for i in range(len(batch['video_ids'])):
             video_id = batch['video_ids'][i]
             # Safe extraction of predicted index
             if pred_indices is not None:
                 try:
                     pred_idx = int(pred_indices[i].item())
                 except Exception:
                     pred_idx = -1
             else:
                 pred_idx = -1
             
             prediction = {
                 'video_id': video_id,
                 'pred_index': pred_idx
             }
             
             # 添加原始ID（如果存在）
             if 'original_ids' in batch and prediction['pred_index'] >= 0:
                try:
                    prediction['pred_id'] = int(batch['original_ids'][i, prediction['pred_index']].item())
                except Exception:
                    prediction['pred_id'] = None
            
             # 添加目标索引（如果存在）并计算rank1_correct
             if 'target_indices' in batch:
                try:
                    true_idx = int(batch['target_indices'][i].item())
                except Exception:
                    true_idx = -1
                prediction['true_index'] = true_idx
                # rank1 correctness
                rank1 = int(prediction['pred_index'] == true_idx) if (prediction['pred_index'] >= 0 and true_idx >= 0) else 0
                prediction['rank1_correct'] = rank1
                # 兼容旧字段（CSV用）
                prediction['correct'] = rank1
            
             # 添加概率分布并计算rank2/rank3（如果需要）
             if probs is not None:
                try:
                    probs_t = torch.as_tensor(probs).cpu()
                    prediction['probabilities'] = probs_t[i].numpy().tolist()
                    # compute topk for this sample
                    topk = torch.topk(probs_t, k=min(3, probs_t.shape[1]), dim=1).indices
                    t_idx = prediction.get('true_index', -1)
                    if t_idx >= 0:
                        pk = topk[i].tolist()
                        prediction['rank2_correct'] = int(t_idx in pk[:2])
                        prediction['rank3_correct'] = int(t_idx in pk[:3])
                    else:
                        prediction['rank2_correct'] = prediction['rank1_correct']
                        prediction['rank3_correct'] = prediction['rank1_correct']
                except Exception:
                    prediction['probabilities'] = []
                    prediction['rank2_correct'] = prediction.get('rank1_correct', 0)
                    prediction['rank3_correct'] = prediction.get('rank1_correct', 0)
             else:
                 prediction['rank2_correct'] = prediction.get('rank1_correct', 0)
                 prediction['rank3_correct'] = prediction.get('rank1_correct', 0)

             # 添加相似度（如果存在），预测/真实解释字段保留为空以兼容Evaluator
             try:
                 similarity = float(losses.get('text_similarities', [0.0])[i].item()) if (losses and 'text_similarities' in losses) else 0.0
             except Exception:
                 similarity = 0.0
             prediction['similarity'] = similarity
             prediction['pred_explanation'] = ''
             prediction['true_explanation'] = ''

             # 添加场景类别（如果存在）
             try:
                 scene = batch.get('scene_categories', [None]*len(batch['video_ids']))[i]
             except Exception:
                 scene = None
             prediction['scene_category'] = scene

             batch_predictions.append(prediction)

         return batch_predictions
    
    def _save_predictions(self, predictions: List[Dict], split: str, epoch: int = None):
        """
        保存预测结果（同时支持CSV和JSON格式）
        
        CSV格式与评估器保持一致，用于可视化和分析
        JSON格式包含更多元信息，用于详细记录
        """
        # 确定文件后缀
        suffix = 'custom' if getattr(self.config, 'model_path', None) else 'best'
        
        # 1. 保存CSV格式（与评估器一致）
        csv_path = os.path.join(self.pred_dir, f'{split}_details_{suffix}.csv')
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, 
                    fieldnames=['video_id', 'correct', 'pred_index', 'true_index', 'similarity'],
                    delimiter='|'
                )
                writer.writeheader()
                for pred in predictions:
                    row = {
                        'video_id': pred['video_id'],
                        'correct': pred.get('correct', ''),
                        'pred_index': pred['pred_index'],
                        'true_index': pred.get('true_index', ''),
                        'similarity': f"{pred.get('text_similarity', 0.0):.4f}"
                    }
                    writer.writerow(row)
            self.logger.info(f"预测结果(CSV)已保存至: {csv_path}")
        except Exception as e:
            self.logger.error(f"保存CSV格式预测结果失败: {str(e)}")
        
        # 2. 保存JSON格式（包含更多信息）
        json_path = os.path.join(self.pred_dir, f'predictions_{split}_{suffix}.json')
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                metadata = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_info': {
                        'model_type': self.model.__class__.__name__,
                        'model_path': getattr(self.config, 'model_path', 'default_best_model'),
                        'is_custom_model': bool(getattr(self.config, 'model_path', None))
                    },
                    'device_info': {
                        'device': str(self.device),
                        'use_amp': self.use_amp,
                        'multi_gpu': self.is_distributed
                    },
                    'performance_stats': self.performance_stats
                }
                
                # 添加模型性能信息（如果有）
                if hasattr(self, 'best_metrics'):
                    metadata['model_info'].update({
                        'best_epoch': self.best_epoch,
                        'best_rank1_accuracy': self.best_rank1_accuracy,
                        'best_loss': self.best_loss,
                        'metrics': self.best_metrics
                    })
                
                json.dump({
                    'predictions': predictions,
                    'metadata': metadata
                }, f, ensure_ascii=False, indent=2)
            self.logger.info(f"预测结果(JSON)已保存至: {json_path}")
        except Exception as e:
            self.logger.error(f"保存JSON格式预测结果失败: {str(e)}")
    def randomize_flow_cnn(self, seed: int = None):
        """在每次重复前重新初始化 flow_cnn 的权重（如果存在）。
        seed: 可选随机种子，用于可重复的随机化。
        返回 True 如果成功执行随机化，False 否则。
        """
        try:
            if seed is not None:
                import random
                random.seed(seed)
                np = __import__('numpy')
                np.random.seed(seed)
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # 获取基础模型（处理 DDP wrapper）
            base_model = getattr(self.model, 'module', self.model)
            # 导航到 dynamic_extractor.optical_flow_encoder.flow_cnn
            if not hasattr(base_model, 'dynamic_extractor'):
                self.logger.warning('模型不包含 dynamic_extractor，无法重置 flow_cnn')
                return False
            dyn = base_model.dynamic_extractor
            of_enc = getattr(dyn, 'optical_flow_encoder', None)
            if of_enc is None or not hasattr(of_enc, 'flow_cnn'):
                self.logger.warning('模型不包含 optical_flow_encoder.flow_cnn，跳过随机化')
                return False

            # 对 flow_cnn 的子模块进行初始化
            for m in of_enc.flow_cnn.modules():
                import torch.nn.init as init
                if isinstance(m, torch.nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif isinstance(m, torch.nn.Linear):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif isinstance(m, torch.nn.LayerNorm):
                    if hasattr(m, 'weight') and m.weight is not None:
                        init.ones_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.zeros_(m.bias)

            self.logger.info('flow_cnn 已随机初始化')
            return True
        except Exception as e:
            self.logger.error(f'flow_cnn 随机化失败: {e}')
            return False