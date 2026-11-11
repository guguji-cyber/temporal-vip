"""
Visualization tool module
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.signal import savgol_filter
import pandas as pd
import csv
from collections import defaultdict
import torch.distributed as dist
import gc


class Visualizer:
    """Visualization tool class"""
    
    def __init__(self, config):
        """
        Initialize visualization tool

        Args:
            config: Configuration object
        """
        self.config = config
        self.vis_dir = config.vis_dir
        self.static_names = config.static_extractor['feature_names']
        self.dynamic_names = config.dynamic_extractor['feature_names']
        self.fusion_names = getattr(config, 'fusion_names', ['aligned', 'static', 'dynamic'])
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.is_main_process = self.rank == 0
        
        # Set plotting style
        plt.style.use('default')
        
        # Initialize history records
        self.history = {
            'losses': [],
            'metrics': {},
            'feature_weights': {},
            'train_records': [],
            'val_records': [],
            'static_weights': [],
            'dynamic_weights': [],
            'fusion_weights': []
        }
        
        # Set visualization style
        sns.set(style='whitegrid', font_scale=1.2)
    
    def plot_loss_from_records(self, records_list, save_path=None):
        """
        直接用内存中的records_list绘制loss曲线，优先画total_loss，没有则画loss/avg_loss。
        """
        x = [int(r['epoch']) + 1 for r in records_list]
        y = [float(r.get('total_loss', r.get('loss', r.get('avg_loss', 0.0)))) for r in records_list]
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(x, y, label='Loss', color='royalblue', linewidth=2, marker='o')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Loss Curve', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.vis_dir, 'loss_curve.png')

        # 只有rank 0保存图片
        if self.is_main_process:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_from_records(self, records_list, save_path=None):
        """
        用rank1_accuracy字段绘制准确率曲线。
        """
        x = [int(r['epoch']) + 1 for r in records_list]
        y = [float(r.get('rank1_accuracy', 0.0)) for r in records_list]
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(x, y, label='Rank-1 Accuracy', color='seagreen', linewidth=2, marker='o')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Rank-1 Accuracy', fontsize=14)
        plt.title('Rank-1 Accuracy Curve', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.vis_dir, 'rank1_accuracy_curve.png')

        # 只有rank 0保存图片
        if self.is_main_process:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def plot_feature_weights_from_records(self, weights_list, weight_type='static', save_path=None):
        """
        动态适配权重分支名，自动绘制所有分支曲线。增强线条区分度。
        weight_type: 'static'/'dynamic'/'fusion'
        """
        if not weights_list:
            return
        # 动态获取分支名
        if weight_type == 'fusion':
            names = self.fusion_names
        else:
            names = list(weights_list[0].keys())
        names = [n for n in names if n != 'epoch']
        if weight_type == 'static':
            default_name = 'static_weights.png'
            title = 'Static Feature Weights Over Epochs'
        elif weight_type == 'dynamic':
            default_name = 'dynamic_weights.png'
            title = 'Dynamic Feature Weights Over Epochs'
        else:
            default_name = 'fusion_weights.png'
            title = 'Fusion Feature Weights Over Epochs'
        x = [int(w['epoch']) + 1 for w in weights_list]
        plt.figure(figsize=(8, 5), dpi=200)
        
        # 线型和标记样式列表
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x', '+']
        color_palette = sns.color_palette("tab10", len(names))
        
        for i, name in enumerate(names):
            y = [float(w.get(name, 0.0)) for w in weights_list]
            # 使用不同的线型、标记和颜色组合
            line_style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            plt.plot(x, y, marker=marker, markersize=7, linestyle=line_style, 
                     linewidth=2, label=name, color=color_palette[i], alpha=0.9)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Weight Value', fontsize=14)
        plt.legend(title='Feature', fontsize=12, loc='best', framealpha=0.8, 
                   edgecolor='gray', ncol=min(3, len(names)))
        plt.grid(True, linestyle='--', alpha=0.6)
        sns.despine()
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.vis_dir, default_name)

        # 只有rank 0保存图片
        if self.is_main_process:
            plt.savefig(save_path, dpi=200)
        plt.close()
    
    def plot_training_curves(self, csv_path):
        """
        绘制损失和准确率曲线
        """
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        loss_col = 'total_loss' if 'total_loss' in df.columns else ('loss' if 'loss' in df.columns else None)
        if loss_col:
            plt.plot(df['epoch']+1, df[loss_col], label='Loss', color='tab:red', marker='o')
        plt.plot(df['epoch']+1, df['accuracy'], label='Accuracy', color='tab:blue', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss & Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'loss_acc_curve.png'), dpi=200)
        plt.close()
    
    def plot_rank_accuracies_from_records(self, records_list, save_path=None):
        """
        绘制rank1、rank2、rank3准确率曲线，使用不同线型和标记来区分重叠线条
        """
        x = [int(r['epoch']) + 1 for r in records_list]
        y1 = [float(r.get('rank1_accuracy', 0.0)) for r in records_list]
        y2 = [float(r.get('rank2_accuracy', 0.0)) for r in records_list]
        y3 = [float(r.get('rank3_accuracy', 0.0)) for r in records_list]
        plt.figure(figsize=(8, 5), dpi=200)
        
        # 使用不同的线型、颜色和标记样式
        plt.plot(x, y1, label='Rank-1', color='royalblue', linewidth=2, marker='o', markersize=8, linestyle='-')
        plt.plot(x, y2, label='Rank-2', color='orange', linewidth=2, marker='s', markersize=8, linestyle='--')
        plt.plot(x, y3, label='Rank-3', color='green', linewidth=2, marker='^', markersize=8, linestyle=':')
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Rank-1/2/3 Accuracy Curve', fontsize=16)
        plt.legend(fontsize=12, loc='best', framealpha=0.8, edgecolor='gray')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.vis_dir, 'rank_accuracies_curve.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def record_epoch_metrics(self, epoch, records, csv_path=None, mode='train', is_main_process=True, details_list=None, class_key=None):
        """
        记录每个epoch的主指标和权重，自动写入csv，支持训练/验证模式。
        动态适配权重分支名和数量。
        """
        if not is_main_process:
            return
        import csv, os
        base_dir = getattr(self.config, 'ID_model_records', self.vis_dir)
        if csv_path is None:
            csv_path = os.path.join(base_dir, f'{mode}_metrics.csv')
        loss_keys = ['cls_loss', 'text_loss', 'contrastive_loss', 'reg_loss', 'ranking_loss']
        row = {}
        row['epoch'] = int(epoch)
        # 优先写入rank准确率
        if 'rank1_accuracy' in records:
            row['rank1_accuracy'] = float(records.get('rank1_accuracy', 0.0))
        if 'rank2_accuracy' in records:
            row['rank2_accuracy'] = float(records.get('rank2_accuracy', 0.0))
        if 'rank3_accuracy' in records:
            row['rank3_accuracy'] = float(records.get('rank3_accuracy', 0.0))
        row['accuracy'] = float(records.get('avg_accuracy', records.get('accuracy', 0.0)) or 0.0)
        row['loss'] = float(records.get('total_loss', records.get('loss', records.get('avg_loss', 0.0))) or 0.0)
        for k in loss_keys:
            v = records.get(k, None)
            if v is not None:
                try:
                    row[k] = float(v) if isinstance(v, (int, float, np.floating)) else 0.0
                except Exception:
                    row[k] = 0.0
        for k, v in records.items():
            if k not in row and ('loss' in k):
                try:
                    row[k] = float(v) if isinstance(v, (int, float, np.floating)) else 0.0
                except Exception:
                    pass
        text_keys = ['pred_explanations', 'explanation_batch_loss', 'explanation_batch_acc', 'explanation_mean_losses', 'explanation_info', 'explanation_result']
        for k in text_keys:
            if k in records:
                row[k] = records[k]
        for k, v in records.items():
            if k not in row and ('acc' in k or 'accuracy' in k):
                try:
                    row[k] = float(v)
                except Exception:
                    row[k] = v
        # 统一写入三类分支权重
        if 'static_branch_avg' in records:
            for i, val in enumerate(records['static_branch_avg']):
                row[f'static_{i}'] = float(val)
        if 'dynamic_branch_avg' in records:
            for i, val in enumerate(records['dynamic_branch_avg']):
                row[f'dynamic_{i}'] = float(val)
        if 'modality_branch_avg' in records:
            for i, val in enumerate(records['modality_branch_avg']):
                row[f'fusion_{i}'] = float(val)
        fieldnames = list(row.keys())
        file_exists = os.path.exists(csv_path)
        if file_exists:
            import pandas as pd
            df = pd.read_csv(csv_path)
            rows = [r for i, r in df.iterrows() if int(r['epoch']) != int(epoch)]
        else:
            rows = []
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                formatted_r = {}
                for k, v in r.items():
                    if k == 'epoch':
                        formatted_r[k] = str(int(float(v)))
                    elif isinstance(v, float):
                        formatted_r[k] = f'{v:.4f}'
                    else:
                        try:
                            if isinstance(v, (np.floating,)):
                                formatted_r[k] = f'{float(v):.4f}'
                            else:
                                formatted_r[k] = v
                        except Exception:
                            formatted_r[k] = v
                writer.writerow({k: formatted_r.get(k, '') for k in fieldnames})
            formatted_row = {}
            for k, v in row.items():
                if k == 'epoch':
                    formatted_row[k] = str(int(v))
                elif isinstance(v, float):
                    formatted_row[k] = f'{v:.4f}'
                else:
                    try:
                        if isinstance(v, (np.floating,)):
                            formatted_row[k] = f'{float(v):.4f}'
                        else:
                            formatted_row[k] = v
                    except Exception:
                        formatted_row[k] = v
            writer.writerow(formatted_row)
        self.save_feature_weights(epoch, records)
        if details_list is not None and len(details_list) > 0:
            self.save_details_list(details_list, epoch, mode=mode)
        if mode == 'train':
            self.history['train_records'].append(dict(row))
        elif mode == 'val':
            self.history['val_records'].append(dict(row))
            # 自动绘制rank准确率曲线
            self.plot_rank_accuracies_from_records(self.history['val_records'])
            # 新增：每个epoch统计并保存各类别rank1/2/3
            if details_list is not None:
                class_csv_dir = os.path.join(self.config.ID_model_records, 'classwise')
                class_vis_dir = os.path.join(self.vis_dir, 'classwise')
                self.save_classwise_epoch_metrics(epoch, details_list, class_csv_dir)
                self.plot_classwise_rank_accuracies_from_csv(class_csv_dir, class_vis_dir)
        # 权重历史收集
        def _append_weight_history(key, contrib, names):
            if not isinstance(contrib, (list, np.ndarray)):
                return
            weight_row = {'epoch': int(epoch)}
            for i, name in enumerate(names):
                try:
                    weight_row[name] = float(contrib[i]) if i < len(contrib) else 0.0
                except Exception:
                    weight_row[name] = 0.0
            self.history[key].append(weight_row)
        # 动态获取分支名
        static_names = getattr(self.config, 'static_extractor', {}).get('feature_names', None)
        dynamic_names = getattr(self.config, 'dynamic_extractor', {}).get('feature_names', None)
        fusion_names = getattr(self.config, 'fusion_names', None)
        # fallback: 用branch_avg
        if not static_names and 'static_branch_avg' in records:
            static_names = [f'static_{i}' for i in range(len(records['static_branch_avg']))]
        if not dynamic_names and 'dynamic_branch_avg' in records:
            dynamic_names = [f'dynamic_{i}' for i in range(len(records['dynamic_branch_avg']))]
        if not fusion_names and 'modality_branch_avg' in records:
            fusion_names = [f'fusion_{i}' for i in range(len(records['modality_branch_avg']))]
        if 'static_branch_avg' in records:
            if 'static_weights' not in self.history:
                self.history['static_weights'] = []
            _append_weight_history('static_weights', records['static_branch_avg'], static_names)
        if 'dynamic_branch_avg' in records:
            if 'dynamic_weights' not in self.history:
                self.history['dynamic_weights'] = []
            _append_weight_history('dynamic_weights', records['dynamic_branch_avg'], dynamic_names)
        if 'modality_branch_avg' in records:
            if 'fusion_weights' not in self.history:
                self.history['fusion_weights'] = []
            _append_weight_history('fusion_weights', records['modality_branch_avg'], fusion_names)
        self.plot_all_curves_from_records(
            self.history['train_records'],
            self.history['val_records'],
            self.history['static_weights'],
            self.history['dynamic_weights'],
            self.history['fusion_weights']
        )
    
    def plot_attention_map(self, attention_weights, labels=None, title='Attention Map', save_path=None):
        """
        绘制注意力热力图
        
        参数:
            attention_weights: 注意力权重矩阵
            labels: 标签列表
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        # 如果是张量，转换为NumPy数组
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # 如果是多头注意力，取平均值
        if attention_weights.ndim > 2:
            attention_weights = np.mean(attention_weights, axis=0)
        
        # 绘制热力图
        sns.heatmap(attention_weights, annot=False, cmap='viridis', xticklabels=labels, yticklabels=labels)
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, classes=None, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
        """
        绘制混淆矩阵
        
        参数:
            cm: 混淆矩阵
            classes: 类别标签
            normalize: 是否归一化
            title: 图表标题
            cmap: 颜色映射
            save_path: 保存路径
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        
        if classes:
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_feature_weights(self, epoch, records):
        """
        保存静态、动态、三模态权重到records/weights目录，字段顺序严格按config或动态推断，全部保留4位小数。
        """
        import csv, os
        base_dir = getattr(self.config, 'ID_model_records', self.vis_dir)
        weights_dir = os.path.join(base_dir, '../weights')
        weights_dir = os.path.abspath(weights_dir)
        # 只有rank 0创建目录
        if self.is_main_process:
            os.makedirs(weights_dir, exist_ok=True)
        def _save_weights_csv(contrib, names, fname):
            if not isinstance(contrib, (list, np.ndarray)):
                return
            path = os.path.join(weights_dir, fname)
            row = {'epoch': int(epoch)}
            for i, name in enumerate(names):
                try:
                    row[name] = float(contrib[i]) if i < len(contrib) else 0.0
                except Exception:
                    row[name] = 0.0
            self._append_row_to_csv(path, row)
        # 动态获取分支名
        static_names = getattr(self.config, 'static_extractor', {}).get('feature_names', None)
        dynamic_names = getattr(self.config, 'dynamic_extractor', {}).get('feature_names', None)
        fusion_names = getattr(self.config, 'fusion_names', None)
        # fallback: 用branch_avg
        if not static_names and 'static_branch_avg' in records:
            static_names = [f'static_{i}' for i in range(len(records['static_branch_avg']))]
        if not dynamic_names and 'dynamic_branch_avg' in records:
            dynamic_names = [f'dynamic_{i}' for i in range(len(records['dynamic_branch_avg']))]
        if not fusion_names and 'modality_branch_avg' in records:
            fusion_names = [f'fusion_{i}' for i in range(len(records['modality_branch_avg']))]
        if 'static_branch_avg' in records:
            _save_weights_csv(records['static_branch_avg'], static_names, 'static_feature_weights.csv')
        if 'dynamic_branch_avg' in records:
            _save_weights_csv(records['dynamic_branch_avg'], dynamic_names, 'dynamic_feature_weights.csv')
        if 'modality_branch_avg' in records:
            _save_weights_csv(records['modality_branch_avg'], fusion_names, 'fusion_feature_weights.csv')

    def _append_row_to_csv(self, csv_path, row):
        """
        辅助方法：将一行数据追加到csv文件，自动补header。
        """
        file_exists = os.path.exists(csv_path)
        fieldnames = list(row.keys())
        # 只有rank 0写入文件
        if self.is_main_process:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    def save_details_list(self, details_list, epoch, mode='val'):
        """
        保存评估详情列表到csv文件（每轮一个文件，合并所有batch），字段顺序为video_id, correct, pred_index, true_index, similarity，用'|'分隔，内容无重复。
        文件路径: logs/preds_logs/{mode}_details_epoch{epoch}.csv
        """
        base_dir = getattr(self.config, 'log_dir', './logs')
        preds_dir = os.path.join(base_dir, 'preds_logs')
        # 只有rank 0创建目录
        if self.is_main_process:
            os.makedirs(preds_dir, exist_ok=True)
        file_path = os.path.join(preds_dir, f'{mode}_details_epoch{epoch}.csv')
        if not details_list:
            return
        # 去重（以video_id为主）
        seen = set()
        unique_details = []
        for d in details_list:
            vid = d.get('video_id', None)
            if vid is not None and vid not in seen:
                seen.add(vid)
                unique_details.append(d)
        # 只有rank 0写入文件
        if self.is_main_process:
            # 字段顺序
            fieldnames = ['video_id', 'correct', 'pred_index', 'true_index', 'similarity']
            # 用'|'分隔写入
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
                writer.writeheader()
                for row in unique_details:
                    formatted_row = {k: row.get(k, '') for k in fieldnames}
                    # similarity保留4位小数
                    if 'similarity' in formatted_row and formatted_row['similarity'] != '':
                        try:
                            formatted_row['similarity'] = f'{float(formatted_row["similarity"]):.4f}'
                        except Exception:
                            pass
                    writer.writerow(formatted_row)

    def plot_loss_and_val_from_records(self, train_records, val_records, save_path=None):
        """
        在同一张图中绘制训练和验证的loss曲线，优先画total_loss，没有则画loss/avg_loss。
        """
        x_train = [int(r['epoch']) + 1 for r in train_records]
        y_train = [float(r.get('total_loss', r.get('loss', r.get('avg_loss', 0.0)))) for r in train_records]
        x_val = [int(r['epoch']) + 1 for r in val_records]
        y_val = [float(r.get('total_loss', r.get('loss', r.get('avg_loss', 0.0)))) for r in val_records]
        plt.figure(figsize=(8, 5), dpi=200)
        
        # 使用不同线型和标记增强视觉区分度
        plt.plot(x_train, y_train, label='Train Loss', color='royalblue', marker='o', 
                 markersize=7, linestyle='-', linewidth=2)
        plt.plot(x_val, y_val, label='Val Loss', color='orange', marker='s', 
                 markersize=7, linestyle='--', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Train & Val Loss Curve', fontsize=16)
        plt.legend(fontsize=12, loc='best', framealpha=0.8, edgecolor='gray')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.vis_dir, 'loss_curve.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    def plot_accuracy_and_val_from_records(self, train_records, val_records, save_path=None):
        """
        用rank1_accuracy字段绘制训练和验证准确率曲线。
        """
        x_train = [int(r['epoch']) + 1 for r in train_records]
        y_train = [float(r.get('rank1_accuracy', 0.0)) for r in train_records]
        x_val = [int(r['epoch']) + 1 for r in val_records]
        y_val = [float(r.get('rank1_accuracy', 0.0)) for r in val_records]
        plt.figure(figsize=(8, 5), dpi=200)
        
        # 使用不同线型和标记增强视觉区分度
        plt.plot(x_train, y_train, label='Train Rank-1 Acc', color='seagreen', 
                 marker='o', markersize=7, linestyle='-', linewidth=2)
        plt.plot(x_val, y_val, label='Val Rank-1 Acc', color='darkorange', 
                 marker='s', markersize=7, linestyle='--', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Rank-1 Accuracy', fontsize=14)
        plt.title('Train & Val Rank-1 Accuracy Curve', fontsize=16)
        plt.legend(fontsize=12, loc='best', framealpha=0.8, edgecolor='gray')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.vis_dir, 'rank1_accuracy_curve.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    def plot_all_curves_from_records(self, train_records_list, val_records_list, static_weights_list, dynamic_weights_list, fusion_weights_list):
        """
        批量绘制并保存所有主指标和权重曲线。
        loss/accuracy为训练和验证同图。
        """
        # Loss曲线
        if train_records_list or val_records_list:
            self.plot_loss_and_val_from_records(train_records_list, val_records_list, save_path=os.path.join(self.vis_dir, 'loss_curve.png'))
        # Accuracy曲线
        if train_records_list or val_records_list:
            self.plot_accuracy_and_val_from_records(train_records_list, val_records_list, save_path=os.path.join(self.vis_dir, 'rank1_accuracy_curve.png'))
        # 权重曲线
        static_weights_list = static_weights_list if static_weights_list is not None else self.history.get('static_weights', [])
        dynamic_weights_list = dynamic_weights_list if dynamic_weights_list is not None else self.history.get('dynamic_weights', [])
        fusion_weights_list = fusion_weights_list if fusion_weights_list is not None else self.history.get('fusion_weights', [])
        if static_weights_list and isinstance(static_weights_list, list) and len(static_weights_list) > 0:
            self.plot_feature_weights_from_records(static_weights_list, 'static', save_path=os.path.join(self.vis_dir, 'static_weights.png'))
        if dynamic_weights_list and isinstance(dynamic_weights_list, list) and len(dynamic_weights_list) > 0:
            self.plot_feature_weights_from_records(dynamic_weights_list, 'dynamic', save_path=os.path.join(self.vis_dir, 'dynamic_weights.png'))
        if fusion_weights_list and isinstance(fusion_weights_list, list) and len(fusion_weights_list) > 0:
            self.plot_feature_weights_from_records(fusion_weights_list, 'fusion', save_path=os.path.join(self.vis_dir, 'fusion_weights.png'))

    def save_classwise_epoch_metrics(self, epoch, details_list, save_dir):
        """
        统计本epoch每个类别的rank1/2/3准确率，追加保存到csv。
        """
        from collections import defaultdict
        os.makedirs(save_dir, exist_ok=True)
        class_samples = defaultdict(list)
        for row in details_list:
            cls = row.get('scene_category', 'unknown')
            class_samples[cls].append(row)
        for cls, samples in class_samples.items():
            total = len(samples)
            if total == 0:
                continue
            correct = sum(int(s['rank1_correct']) for s in samples)
            rank1 = correct / total
            rank2 = sum(int(s.get('rank2_correct', s['rank1_correct'])) for s in samples) / total
            rank3 = sum(int(s.get('rank3_correct', s['rank1_correct'])) for s in samples) / total
            csv_path = os.path.join(save_dir, f'{cls}_metrics.csv')
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                fieldnames = ['epoch', 'rank1_accuracy', 'rank2_accuracy', 'rank3_accuracy', 'rank1_correct_count', 'total_count']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'epoch': epoch,
                    'rank1_accuracy': f'{rank1:.6f}',
                    'rank2_accuracy': f'{rank2:.6f}',
                    'rank3_accuracy': f'{rank3:.6f}',
                    'rank1_correct_count': correct,
                    'total_count': total
                })
    def plot_classwise_rank_accuracies_from_csv(self, save_dir, vis_dir):
        """
        读取每个类别的csv，绘制rank1/2/3曲线。
        """
        import os
        import pandas as pd
        os.makedirs(vis_dir, exist_ok=True)
        for fname in os.listdir(save_dir):
            if not fname.endswith('_metrics.csv'):
                continue
            cls = fname.replace('_metrics.csv', '')
            csv_path = os.path.join(save_dir, fname)
            df = pd.read_csv(csv_path)
            x = df['epoch'] + 1
            y1 = df['rank1_accuracy']
            y2 = df['rank2_accuracy']
            y3 = df['rank3_accuracy']
            plt.figure(figsize=(8, 5), dpi=200)
            
            # 使用不同线型和标记提高可区分度
            plt.plot(x, y1, label='Rank-1', color='royalblue', linewidth=2, 
                     marker='o', markersize=8, linestyle='-')
            plt.plot(x, y2, label='Rank-2', color='orange', linewidth=2, 
                     marker='s', markersize=8, linestyle='--')
            plt.plot(x, y3, label='Rank-3', color='green', linewidth=2, 
                     marker='^', markersize=8, linestyle=':')
            
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.title(f'Rank-1/2/3 Accuracy Curve ({cls})', fontsize=16)
            plt.legend(fontsize=12, loc='best', framealpha=0.8, edgecolor='gray')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            save_path = os.path.join(vis_dir, f'{cls}_rank_accuracies_curve.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close() 