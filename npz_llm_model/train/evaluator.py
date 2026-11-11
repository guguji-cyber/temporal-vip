"""
Evaluator module for model evaluation
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.amp import autocast
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
import traceback
import openpyxl
from openpyxl.styles import Font
import csv
from sentence_transformers import SentenceTransformer, util
import gc

# 恢复相对导入
from ..utils.logger import Logger
from ..utils.metrics import Metrics, normalize_metrics
from ..utils.visualization import Visualizer
from ..train.losses import TotalLoss

class Evaluator:
    """Evaluator class for model evaluation"""
    
    def __init__(self, model, data_loader, config):
        """
        初始化评估管理器
        
        参数:
            model: 模型对象
            data_loader: 数据加载器对象
            config: 配置对象
        """
        self.config = config
        self.model = model
        self.data_loader = data_loader
        
        # 初始化日志记录器和可视化工具
        self.logger = Logger(config)
        self.visualizer = Visualizer(config)
        
        # 初始化指标计算器
        self.metrics = Metrics()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 分布式训练设置
        self.is_distributed = dist.is_initialized()
        
        # 设置损失函数
        self.criterion = TotalLoss(
            cls_weight=getattr(config, 'cls_loss_weight', 1.0),
            text_weight=getattr(config, 'text_loss_weight', 0.5),
            contrastive_weight=getattr(config, 'contrastive_loss_weight', 0.3),
            reg_weight=getattr(config, 'reg_weight', 0.0001),
            config=config
        )
        
        # 损失函数
        self.cls_loss_fn = nn.CrossEntropyLoss()
        
        self.text_total_loss = 0.0
        self.text_total_acc = 0.0
        self.text_batches = 0
    
    def evaluate(self, epoch=None, save_results=True):
        """
        评估模型性能
        参数:
            epoch: 当前训练的epoch，可选
            save_results: 是否保存评估结果
        返回:
            Dict: 评估指标
        """
        # 分布式环境下所有进程都评估
        is_distributed = self.is_distributed
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'cls_loss': 0.0,
            'text_loss': 0.0,
            'contrastive_loss': 0.0,
            'reg_loss': 0.0,
            'text_similarity': 0.0,
            'ranking_loss': 0.0,
            'total_pairs': 0,
        }
        total_count = 0
        correct_total = 0
        details_list = []
        val_loader = self.data_loader.get_val_loader()
        eval_iter = tqdm(val_loader, desc="Evaluating", dynamic_ncols=True, disable=(rank != 0))
        all_pred_indices = []
        all_target_indices = []
        all_topk_indices = []
        pred_true_list = []  # 新增：用于收集所有(video_id, pred_id, true_id)
        explanation_rows = []  # 新增：用于保存解释csv
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_iter):
                try:
                    batch_data = self._move_to_device(batch_data)
                    if batch_data is None:
                        continue
                    with autocast('cuda'):
                        outputs = self.model(batch_data, criterion=self.criterion)
                    if outputs is None:
                        continue
                    pred_indices = outputs.get('pred_indices', None)
                    probabilities = outputs.get('probabilities', None)
                    target_indices = batch_data.get('target_indices', None)
                    if pred_indices is None or target_indices is None:
                        continue

                    if probabilities is not None:
                        _, topk_indices = torch.topk(probabilities, 3, dim=1)
                        all_topk_indices.append(topk_indices.cpu())

                    if (not self.is_distributed) or (dist.get_rank() == 0):
                        for i in range(pred_indices.size(0)):
                            pred_true_list.append((batch_data['video_ids'][i], pred_indices[i].item(), target_indices[i].item()))
                    all_pred_indices.append(pred_indices.cpu())
                    all_target_indices.append(target_indices.cpu())
                    similarities = outputs.get('losses', {}).get('text_similarities', [])
                    losses = outputs.get('losses', None)
                    batch_size = pred_indices.size(0)
                    pred_explanations = outputs.get('pred_explanations', [])
                    true_explanations = batch_data.get('vip_explanations', [])

                    # 处理pred_explanations的格式，可能是字典列表或字符串列表
                    if pred_explanations is None or len(pred_explanations) == 0:
                        pred_explanations = [""] * batch_size
                    else:
                        # 如果是字典列表，提取text字段
                        if isinstance(pred_explanations[0], dict) and 'text' in pred_explanations[0]:
                            pred_explanations = [exp['text'] for exp in pred_explanations]
                        # 确保长度匹配batch_size
                        if len(pred_explanations) < batch_size:
                            pred_explanations.extend([""] * (batch_size - len(pred_explanations)))

                    if true_explanations is None or len(true_explanations) == 0:
                        true_explanations = [""] * batch_size

                    # 处理similarities，确保它是list而不是tensor
                    if similarities is None:
                        similarities = [0.0] * batch_size
                    elif isinstance(similarities, torch.Tensor):
                        similarities = similarities.cpu().tolist()
                    elif len(similarities) == 0:
                        similarities = [0.0] * batch_size

                    if losses is not None:
                        val_metrics['total_loss'] += float(losses.get('total_loss', 0.0)) * batch_size
                        val_metrics['cls_loss'] += float(losses.get('classification_loss', 0.0)) * batch_size
                        val_metrics['text_loss'] += float(losses.get('text_similarity_loss', 0.0)) * batch_size
                        val_metrics['contrastive_loss'] += float(losses.get('contrastive_loss', 0.0)) * batch_size
                        val_metrics['reg_loss'] += float(losses.get('regularization_loss', 0.0)) * batch_size

                        # 检查similarities是否有效（使用len而不是直接布尔判断）
                        if len(similarities) == 0 or all(s == 0.0 for s in similarities):
                            # 如果没有有效的相似度分数，跳过文本相似度累加
                            pass
                        else:
                            val_metrics['text_similarity'] += sum(similarities)
                        # 统计pair数
                        num_pairs = 0
                        if 'ranking_scores' in losses and losses['ranking_scores'] is not None:
                            score1, score2, ranking_target = losses['ranking_scores']
                            num_pairs = int(score1.numel())
                        val_metrics['ranking_loss'] += float(losses.get('ranking_loss', 0.0)) * num_pairs
                        val_metrics['total_pairs'] += num_pairs
                    total_count += batch_size
                    correct = (pred_indices == target_indices).sum().item()
                    correct_total += correct
                    # 获取当前批次的topk结果
                    current_topk = None
                    if probabilities is not None:
                        _, current_topk = torch.topk(probabilities, 3, dim=1)  # [batch_size, 3]

                    for i in range(batch_size):
                        val = similarities[i] if i < len(similarities) else 0.0
                        sim = val.item() if hasattr(val, 'item') else float(val)
                        pred_exp = pred_explanations[i] if i < len(pred_explanations) else ""
                        true_exp = true_explanations[i] if i < len(true_explanations) else ""
                        scene_category = batch_data.get('scene_categories', [None]*batch_size)[i] if 'scene_categories' in batch_data else None

                        # 计算rank2/rank3正确性
                        target = target_indices[i].item()
                        if current_topk is not None:
                            topk = current_topk[i].tolist()
                            rank2_correct = int(target in topk[:2])
                            rank3_correct = int(target in topk[:3])
                        else:
                            # 如果没有topk结果，只能用rank1结果
                            rank2_correct = int(pred_indices[i] == target)
                            rank3_correct = int(pred_indices[i] == target)

                        row = {
                            'video_id': batch_data['video_ids'][i],
                            'pred_index': pred_indices[i].item(),
                            'true_index': target,
                            'rank1_correct': int(pred_indices[i] == target),
                            'rank2_correct': rank2_correct,
                            'rank3_correct': rank3_correct,
                            'pred_explanation': pred_exp,
                            'true_explanation': true_exp,
                            'similarity': sim,
                            'scene_category': scene_category
                        }
                        explanation_rows.append(row)
                        details_list.append(row)
                    avg_loss = val_metrics['total_loss'] / total_count if total_count > 0 else 0.0
                    avg_similarity = val_metrics['text_similarity'] / total_count if total_count > 0 else 0.0
                    avg_acc = correct_total / total_count if total_count > 0 else 0.0
                    eval_iter.set_postfix({
                        'acc': f"{avg_acc:.4f}",
                        'loss': f"{avg_loss:.4f}",
                        'txt_sim': f"{avg_similarity:.4f}"
                    })
                except Exception as e:
                    error_msg = f"评估批次 {batch_idx} 时出错: {str(e)}\n"
                    error_msg += traceback.format_exc()
                    print(error_msg)
                    continue
        if is_distributed:
            all_pred_indices = torch.cat(all_pred_indices, dim=0) if all_pred_indices else torch.empty(0, dtype=torch.long)
            all_target_indices = torch.cat(all_target_indices, dim=0) if all_target_indices else torch.empty(0, dtype=torch.long)
            all_topk_indices = torch.cat(all_topk_indices, dim=0) if all_topk_indices else torch.empty(0, 3, dtype=torch.long)

            all_pred_indices = all_pred_indices.to(self.device)
            all_target_indices = all_target_indices.to(self.device)
            all_topk_indices = all_topk_indices.to(self.device)

            gather_pred = [torch.empty_like(all_pred_indices) for _ in range(world_size)]
            gather_target = [torch.empty_like(all_target_indices) for _ in range(world_size)]
            gather_topk = [torch.empty_like(all_topk_indices) for _ in range(world_size)]

            dist.all_gather(gather_pred, all_pred_indices)
            dist.all_gather(gather_target, all_target_indices)
            dist.all_gather(gather_topk, all_topk_indices)

            if rank == 0:
                all_pred_indices = torch.cat(gather_pred, dim=0)
                all_target_indices = torch.cat(gather_target, dim=0)
                all_topk_indices = torch.cat(gather_topk, dim=0)

            for k in val_metrics:
                tensor = torch.tensor([val_metrics[k]], dtype=torch.float32, device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                val_metrics[k] = tensor.item()
        else:
            all_pred_indices = torch.cat(all_pred_indices, dim=0) if all_pred_indices else torch.empty(0, dtype=torch.long)
            all_target_indices = torch.cat(all_target_indices, dim=0) if all_target_indices else torch.empty(0, dtype=torch.long)
            all_topk_indices = torch.cat(all_topk_indices, dim=0) if all_topk_indices else torch.empty(0, 3, dtype=torch.long)
        if ((not is_distributed) or (rank == 0)):
            total_count = all_pred_indices.size(0)
            correct_total = 0
            accuracy = 0.0
            rank2_accuracy = 0.0
            rank3_accuracy = 0.0

            if total_count > 0:
                correct_total = (all_pred_indices == all_target_indices).sum().item()
                accuracy = correct_total / total_count
                
                if all_topk_indices.numel() > 0:
                    target_unsqueezed = all_target_indices.unsqueeze(1)
                    matches = all_topk_indices.eq(target_unsqueezed)
                    
                    correct_rank2 = matches[:, :2].any(dim=1).sum().item()
                    correct_rank3 = matches[:, :3].any(dim=1).sum().item()
                    
                    rank2_accuracy = correct_rank2 / total_count
                    rank3_accuracy = correct_rank3 / total_count

            val_metrics['total_count'] = total_count
            val_metrics = normalize_metrics(
                val_metrics,
                count_key='total_count',
                exclude_keys=['total_pairs', 'static_branch_avg', 'dynamic_branch_avg', 'modality_branch_avg', 'correct_count', 'total_count']
            )
            val_metrics['accuracy'] = accuracy
            val_metrics['rank2_accuracy'] = rank2_accuracy
            val_metrics['rank3_accuracy'] = rank3_accuracy
            val_metrics['ranking_loss'] = val_metrics['ranking_loss'] / val_metrics['total_pairs'] if val_metrics['total_pairs'] > 0 else 0.0
            val_metrics['total_pairs'] = val_metrics['total_pairs']
            val_metrics['correct_count'] = correct_total
            # 保证有rank1_accuracy字段
            val_metrics['rank1_accuracy'] = val_metrics.get('accuracy', 0.0)
            # 统一交由visualizer收集、写入csv
            self.visualizer.record_epoch_metrics(epoch, val_metrics, mode='val', is_main_process=True, details_list=details_list, class_key='scene_category')
            # 只有rank 0保存解释csv
            if epoch is not None and save_results and ((not is_distributed) or (rank == 0)):
                save_dir = self.config.explanation_generator_records
                # 目录已在config中创建，无需重复创建
                save_path = os.path.join(save_dir, f'epoch_{epoch+1}_val_explanations.csv')
                with open(save_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['video_id', 'pred_index', 'true_index', 'rank1_correct', 'rank2_correct', 'rank3_correct', 'pred_explanation', 'true_explanation', 'similarity', 'scene_category'])
                    writer.writeheader()
                    for row in explanation_rows:
                        writer.writerow(row)

            # 清理评估数据，释放内存
            del explanation_rows, details_list, all_pred_indices, all_target_indices, all_topk_indices
            gc.collect()

            return val_metrics
        else:
            return {} # 非主进程不返回指标
    
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