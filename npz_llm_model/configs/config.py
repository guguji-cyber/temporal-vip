"""
统一配置参数类
"""
import os
from glob import glob
import argparse
from datetime import datetime
import torch
import torch.distributed as dist
import gc

class Config:
    """配置参数类"""
    def __init__(self):
        # 动态获取项目根目录（基于当前工作目录）
        self.base_dir = os.getcwd()
        # 直接使用预处理好的数据目录（使用相对路径）
        self.output_dir = os.path.join(self.base_dir, 'code/src/npz_llm_model/outputs')
        self.npz_dir = os.path.join(self.base_dir, 'code/data/vip/preprocessed_fixed')  # 预处理npz主目录，含train/val/test子目录
        self.json_dir = os.path.join(self.base_dir, 'code/data/vip/llm_marked_videos_description')  # LLM生成的视频描述json目录
        
        # 移除自动生成的时间戳路径（改为在update_from_args中动态生成）
        self.timestamp = None
        self.run_dir = None
        self.log_dir = None
        self.pred_dir = None
        self.checkpoint_dir = None
        self.main_model_dir = None
        self.explanation_generator_dir = None
        self.vis_dir = None
        self.records_dir = None
        self.ID_model_records = None
        self.explanation_generator_records = None
        self.weights_records = None
        
        # 训练参数（保持默认值不变）
        self.mode = 'train'
        self.batch_size = 32
        self.epochs = 50
        self.multi_gpu = False
        self.test_mode = False
        self.test_data_ratio = 1/5
        
        # self.dropout = 0.1
        # === 强力正则化建议：过拟合时建议将dropout调高到0.3 ===
        self.dropout = 0.3

        # self.weight_decay = 1e-5
        # === 强力正则化建议：过拟合时建议将weight_decay调高到1e-4 ===
        self.weight_decay = 1e-4

        # self.learning_rate = 1e-4
        # === 建议：过拟合时可适当降低学习率 ===
        self.learning_rate = 5e-5
        
        self.clip_grad_norm = 1.0
        self.num_workers = 4
        self.device = 'cuda:0'
        self.eval_interval = 1
        self.save_interval = 5
        self.model_path = ''
        
        # 模型参数（保持默认值不变）
        self.hidden_dim = 256
        self.num_heads = 8
        self.num_layers = 4
        self.max_persons = 20
        self.max_frames = 120
        self.frame_height = 192
        self.frame_width = 336
        self.seq_len = 20  # 每个ID的序列长度
        
        # 特征融合专用参数（保持默认值不变）
        self.num_fusion_heads = 8
        self.num_fusion_layers = 3
        
        # 特征提取参数（保持默认值不变）
        self.static_feature_dim = 32  # 静态特征维度
        self.dynamic_feature_dim = 512  # 动态特征维度
        self.text_feature_dim = 512  # 文本特征维度
        
        # 特征交互参数（保持默认值不变）
        self.use_cross_attention = True
        self.use_adaptive_weights = True
        self.use_residual_connection = True
        
        # 多粒度对比学习参数（保持默认值不变）
        self.contrastive_temperature = 0.07  # 温度参数
        self.global_weight = 0.5  # 全局对比学习权重
        self.local_weight = 0.5  # 局部对比学习权重
        self.contrastive_loss_weight = 0.3  # 对比学习总体权重
        
        # 多模态融合参数（保持默认值不变）
        self.fusion_agg_method = 'attention'  # 可选: 'mean', 'max', 'attention'
        self.fusion_type = "transformer"
        
        self.is_ablation = False  # 是否为消融实验，默认关闭
        
        # 增强型融合参数（保持默认值不变）
        self.use_enhanced_fusion = True  # 是否使用增强型融合
        self.modal_gate_temperature = 1.0  # 模态门控温度参数
        self.use_global_fusion_weights = True  # 是否使用全局融合权重
        
        # 文本特征增强参数（保持默认值不变）
        self.use_enhanced_text = True
        self.use_id_mapping = True
        self.bert_feature_dim = 768
        self.text_attention_heads = 4
        self.text_interaction_layers = 2
        self.text_dropout = self.dropout
        self.text_feature_gate_dim = 256
        
        # 特征权重初始比值（保持默认值不变）
        self.centrality_weight = 4
        self.visual_weight = 6  # 替代原来的size_weight和clarity_weight
        self.speech_weight = 8
        self.pose_weight = 5
        self.dynamic_fusion_weight = 7
        self.static_fusion_weight = 6
        self.text_fusion_weight = 4
        self.spatiotemporal_fusion_weight = 5  # 时空特征权重
        self.temperature = 0.5  # 分类预测温度参数
        
        # 文本生成权重（保持默认值不变）
        self.text_generation_weight = 0.3  # 文本生成损失权重
        
        # 解释生成参数（保持默认值不变）
        self.explanation_mode = 'both'  # 可选：'template', 'generative', 'both'
        self.explanation_fusion = {
            'coverage_weight': 0.4,    # 特征覆盖度权重
            'coherence_weight': 0.3,   # 文本连贯性权重
            'relevance_weight': 0.3,   # 特征相关性权重
            'max_length': 250,         # 生成文本最大长度
            'num_beams': 4            # beam search的beam数量
        }
        
        # 损失权重（保持默认值不变）
        self.cls_loss_weight = 1.0
        self.text_loss_weight = 0.5
        self.reg_weight = 0.005
        self.ranking_loss_weight = 0.2
        self.ranking_margin = 0.2
        
        self.debug = False  # 是否输出详细调试信息
        
        # 加载模型（使用code/src内的模型目录）
        models_base = os.path.join(self.base_dir, 'code/src/models/huggingface/hub')

        bart_model_path = os.path.join(models_base, 'models--fnlp--bart-base-chinese/snapshots')
        bart_subdirs = glob(os.path.join(bart_model_path, '*'))
        if not bart_subdirs:
            raise ValueError(f"未找到BART模型: {bart_model_path}")
        self.bart_path = bart_subdirs[0]

        bert_model_path = os.path.join(models_base, 'models--google-bert--bert-base-chinese/snapshots')
        bert_subdirs = glob(os.path.join(bert_model_path, '*'))
        if not bert_subdirs:
            raise ValueError(f"未找到BERT模型: {bert_model_path}")
        self.bert_path = bert_subdirs[0]

        st_model_path = os.path.join(models_base, 'models--sentence-transformers--distiluse-base-multilingual-cased-v1/snapshots')
        st_subdirs = glob(os.path.join(st_model_path, '*'))
        if not st_subdirs:
            raise ValueError(f"未找到ST模型: {st_model_path}")
        self.st_path = st_subdirs[0]
        
        # 特征标签（保持默认值不变）
        self.fusion_names = ['aligned', 'static', 'dynamic']
        
        # 静态特征提取器参数（保持默认值不变）
        self.static_extractor = {
            'feature_names': ['area', 'centrality', 'clarity'],  # 特征类型
            'backbone': 'resnet18',  # 视觉backbone类型
            'temporal_heads': 4,  # 时序注意力头数
            'feature_encoder_hidden': 64,  # 特征编码器隐层维度
            'dropout': self.dropout,  # dropout比率
            'feature_dim': self.static_feature_dim  # 静态特征维度
        }
        
        # 动态特征提取器参数（保持默认值不变）
        self.dynamic_extractor = {
            'feature_names': ['action', 'speech'],  # 特征类型
            'feature_dim': self.dynamic_feature_dim,  # 动态特征维度
            'action_encoder': {
                'backbone': '3d_resnet',  # 动作特征backbone
                'num_layers': 3,  # 3D ResNet层数
                'temporal_heads': 4,  # 时序注意力头数
                'dropout': self.dropout
            },
            'lip_encoder': {
                'input_dim': 40,  # 唇动关键点维度
                'hidden_dim': 64,  # 隐层维度
                'num_layers': 2,  # Transformer层数
                'temporal_heads': 4,  # 时序注意力头数
                'dropout': self.dropout
            },
            # 实验开关：用光流替换时序模块（同时替换 action 与 lip），默认关闭
            'replace_temporal_with_flow': False,
            # 唇动关键点缓存目录（纯加速，数值逐位等价）。
            # 由 baselines/precompute_lip.py 生成；留空则实时跑 MediaPipe（很慢）。
            'lip_cache_dirs': [],
            # 光流后端配置：'farneback'（默认）或 'raft'；若使用 raft，请提供 raft_model_path
            'optical_flow': {
                'backend': 'farneback',  # 'farneback' or 'raft'
                'raft_model_path': '',  # 如果使用RAFT，需提供checkpoint路径
                'raft_device': 'cpu',
                'raft_small': True
            }
        }
        
        # 文本特征提取器参数（保持默认值不变）
        self.text_extractor = {
            'bert_path': self.bert_path,  # BERT模型路径
            'feature_dim': self.text_feature_dim,  # 输出特征维度，唯一引用
            'bert_feature_dim': 768,  # BERT特征维度
            'attention_heads': 4,  # 注意力头数
            'max_length': 250,  # 最大序列长度
            'dropout': self.dropout,  # Dropout比率
            'projector_hidden': 1024,  # 投影层隐藏维度
            'gate_ratio': 0.3,  # 特征门控比例
            'feature_fields': ['location', 'action', 'expression', 'interaction']  # 人物描述字段
        }
        
        # 时空感知对齐参数（保持默认值不变）
        self.temporal_alignment = {
            'fusion_dim': 256,           # 融合特征维度
            'num_heads': 8,              # 注意力头数
            'num_layers': 2,             # TransformerEncoder/Cross-Attention层数
            'dropout': self.dropout,              # Dropout比率
            'use_norm': True,            # 是否使用LayerNorm
            'agg_method': self.fusion_agg_method,   # 特征聚合方式: mean/max/attention
            'use_residual': True,        # 是否使用残差连接 (仅mlp和concat模式可选)
            'temporal_window': 5,        # 时间窗口大小
            'use_motion_prior': True,    # 是否使用运动先验
            'feature_gate': True,        # 是否使用特征门控 (仅gated模式使用)
            'gate_temperature': 0.1,     # 特征门控温度系数
            'fusion_type': self.fusion_type, # 融合方式: concat/mlp/gated/transformer，默认transformer
            # QKV模式选择：static_query 或 dynamic_query
            'qkv_mode': 'static_query'   # 默认QKV模式
        }

        self.use_template_only = True  # 是否只用模板生成重要人物解释文本（True则不走大模型生成）

        # 早停机制参数（默认None，不启用早停，用户可自定义）
        self.early_stop_patience = None

        self.ablation_mode = None
        self.ablate_feature = None

    def update_from_args(self, args):
        """从命令行参数更新配置"""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        
        self.ablate_feature = getattr(args, 'ablate_feature', None)
        if self.ablate_feature:
            print(f"--- ABLATION MODE: Disabling feature '{self.ablate_feature}' ---")
            if self.ablate_feature in self.static_extractor['feature_names']:
                self.static_extractor['feature_names'].remove(self.ablate_feature)
                print(f"Updated static features: {self.static_extractor['feature_names']}")
            if self.ablate_feature in self.dynamic_extractor['feature_names']:
                self.dynamic_extractor['feature_names'].remove(self.ablate_feature)
                print(f"Updated dynamic features: {self.dynamic_extractor['feature_names']}")
        
        # 如果使用光流替换时序模块，更新 feature_names 并打印提示
        if hasattr(args, 'replace_temporal_with_flow') and args.replace_temporal_with_flow:
             print("--- EXPERIMENT MODE: Replacing temporal modules with optical flow features ---")
             # 为了兼容上层期待的长度 (通常为2)，我们用相同名字占两个槽
             self.dynamic_extractor['feature_names'] = ['optical_flow', 'optical_flow']
             self.dynamic_extractor['replace_temporal_with_flow'] = True

        # 读取光流后端相关命令行参数
        if hasattr(args, 'optical_flow_backend') and args.optical_flow_backend is not None:
            self.dynamic_extractor['optical_flow']['backend'] = args.optical_flow_backend
        if hasattr(args, 'raft_model_path') and args.raft_model_path is not None:
            self.dynamic_extractor['optical_flow']['raft_model_path'] = args.raft_model_path
        if hasattr(args, 'raft_device') and args.raft_device is not None:
            self.dynamic_extractor['optical_flow']['raft_device'] = args.raft_device

        # 唇动关键点缓存（纯加速，数值逐位等价）；逗号分隔的目录列表
        if getattr(args, 'lip_cache_dirs', None):
            _dirs = [d.strip() for d in str(args.lip_cache_dirs).split(',') if d.strip()]
            self.dynamic_extractor['lip_cache_dirs'] = _dirs
            print(f"--- Lip cache enabled: {_dirs} ---")

        # 特殊处理 --test 参数
        if args.test is not None:
             # 如果提供了 --test，args.test 会是一个浮点数
             self.test_mode = True
             self.test_data_ratio = args.test # 直接取 float
        else:
             # 如果没有提供 --test，则保持默认值 test_mode=False, test_data_ratio=1/5
             pass
        
        # 标记用户明确设置的参数
        if hasattr(args, 'multi_gpu') and args.multi_gpu:
            self._multi_gpu_explicitly_set = True

        # 处理特征选择和融合类型参数
        if hasattr(args, 'feature_selection') and args.feature_selection:
            self.feature_selection = args.feature_selection.split(',')
            self.temporal_alignment['fusion_type'] = self.fusion_type
        if hasattr(args, 'fusion_type') and args.fusion_type:
            self.fusion_type = args.fusion_type
            self.temporal_alignment['agg_method'] = self.fusion_agg_method
        # QKV模式选择：static_query 或 dynamic_query
        if hasattr(args, 'qkv_mode') and args.qkv_mode:
            self.temporal_alignment['qkv_mode'] = args.qkv_mode
        
        # 延迟时间戳和路径生成，等待分布式初始化完成
        self._directories_created = False
        
        # 新增：命令行参数支持early_stop_patience
        if hasattr(args, 'early_stop') and args.early_stop is not None:
             self.early_stop_patience = args.early_stop

        # 新增：预测重复次数与分割
        if hasattr(args, 'predict_repeats') and args.predict_repeats is not None:
            self.predict_repeats = args.predict_repeats
        else:
            self.predict_repeats = 1
        if hasattr(args, 'predict_split') and args.predict_split is not None:
            self.predict_split = args.predict_split
        else:
            self.predict_split = 'val'
        # 每次重复前将默认随机化 flow_cnn（用于估计随机初始化带来的方差）
        # 不暴露为用户开关，直接采用默认随机化以避免误用
        self.randomize_flow_cnn_per_repeat = True

        # 新增：超参数调优参数覆盖
        if hasattr(args, 'lambda_cont') and args.lambda_cont is not None:
            self.contrastive_loss_weight = args.lambda_cont
            print(f"[Config] 覆盖对比学习损失权重: λ_cont = {args.lambda_cont}")

        if hasattr(args, 'lambda_text') and args.lambda_text is not None:
            self.text_loss_weight = args.lambda_text
            print(f"[Config] 覆盖文本损失权重: λ_text = {args.lambda_text}")

        if hasattr(args, 'lambda_reg') and args.lambda_reg is not None:
            self.reg_weight = args.lambda_reg
            print(f"[Config] 覆盖正则化权重: λ_reg = {args.lambda_reg}")

        if hasattr(args, 'temperature_scale') and args.temperature_scale is not None:
            self.temperature = args.temperature_scale
            print(f"[Config] 覆盖温度参数: temperature = {args.temperature_scale}")
        
    @staticmethod
    def get_args():
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='视频重要人物检测系统')
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], help='运行模式')
        parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
        parser.add_argument('--epochs', type=int, default=1, help='训练轮次')
        parser.add_argument('--num_workers', type=int, default=4, help='DataLoader的worker数量')
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
        parser.add_argument('--multi_gpu', action='store_true', help='启用多GPU训练')
        parser.add_argument('--model_path', type=str, default='', help='预训练模型路径')
        parser.add_argument('--use_enhanced_fusion', action='store_true', default=True, help='是否使用增强型融合')
        parser.add_argument('--contrastive_loss_weight', type=float, default=0.3, help='对比学习损失权重')
        parser.add_argument('--explanation_mode', type=str, default='both', choices=['template', 'generative', 'both'], help='解释生成模式：template-仅模板，generative-仅生成，both-两者都用')
        parser.add_argument('--debug', action='store_true', help='是否详细调试输出')
        # 新增消融实验参数
        parser.add_argument('--feature_selection', type=str, default='static,dynamic,text', help='特征组合，逗号分隔')
        parser.add_argument('--fusion_type', type=str, default='transformer', choices=['concat', 'mlp', 'gated', 'transformer'], help='融合方式: concat/mlp/gated/transformer')
        parser.add_argument('--is_ablation', action='store_true', help='是否为消融实验，仅保存最佳模型')
        # 修改 --test 参数，使其必须接收一个浮点数值
        parser.add_argument('--test', type=float, default=None, help='启用测试模式，并指定使用的数据比例')
        parser.add_argument('--early_stop', type=int, default=None, help='早停耐心轮数（None为不早停，正整数为自定义早停）')
        parser.add_argument('--ablate_feature', type=str, default=None, help="指定要消融的子特征, e.g., 'area', 'clarity', 'action'")
        # 实验开关：替换动作和唇动模块为光流特征
        parser.add_argument('--replace_temporal_with_flow', action='store_true', help='用光流特征替换 action 和 lip 模块')
        parser.add_argument('--optical_flow_backend', type=str, default=None, choices=['farneback', 'raft'], help='选择光流后台: farneback 或 raft')
        parser.add_argument('--raft_model_path', type=str, default=None, help='RAFT 模型 checkpoint 路径（用于 optical_flow_backend=raft）')
        parser.add_argument('--raft_device', type=str, default=None, help='RAFT 推理设备, e.g., cpu or cuda:0')
        parser.add_argument('--lip_cache_dirs', type=str, default=None,
                            help='唇动关键点缓存目录，逗号分隔（由 precompute_lip.py 生成）；不传则实时跑 MediaPipe')

        # 新增超参数调优参数
        parser.add_argument('--lambda_cont', type=float, default=None, help='对比学习损失权重 λ_cont，覆盖默认的contrastive_loss_weight')
        parser.add_argument('--lambda_text', type=float, default=None, help='文本相似度损失权重 λ_text，覆盖默认的text_loss_weight')
        parser.add_argument('--lambda_reg', type=float, default=None, help='L2正则化损失权重 λ_reg，覆盖默认的reg_weight')
        parser.add_argument('--temperature_scale', type=float, default=None, help='分类预测温度参数，覆盖默认的temperature')
        # Predict repeat options: run multiple evaluation repeats (useful for statistics)
        parser.add_argument('--predict_repeats', type=int, default=1, help='重复预测次数（在验证集或测试集上重复多次以统计稳定性）')
        parser.add_argument('--predict_split', type=str, default='val', choices=['val', 'test'], help='进行预测/评估时使用的数据集分割')
        parser.add_argument('--qkv_mode', type=str, default='static_query', choices=['static_query', 'dynamic_query'], help='Cross-Attention QKV 模式: static_query (默认) 或 dynamic_query')

        args = parser.parse_args()
        return args

    def _get_unified_timestamp(self):
        """获取统一时间戳，确保所有DDP进程使用相同的时间戳"""
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                # 只有rank 0生成时间戳
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            else:
                timestamp = ""

            # 使用简单的字符串列表广播
            timestamp_list = [timestamp]
            dist.broadcast_object_list(timestamp_list, src=0)
            timestamp = timestamp_list[0]
        else:
            # 单进程环境：直接生成
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        return timestamp

    def cleanup_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

    def get_memory_info(self):
        """获取内存使用信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
        return "CUDA not available"

    def adapt_config_for_training_mode(self):
        """根据训练环境自动调整配置，但不覆盖用户明确指定的参数"""
        # 检测是否在DDP环境中
        is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
        gpu_count = torch.cuda.device_count()

        # 记录原始值，用于判断是否被用户修改过
        original_batch_size = self.batch_size
        original_num_workers = self.num_workers

        if is_ddp and gpu_count > 1:
            # DDP模式：只在必要时调整，不强制覆盖用户设置
            if not hasattr(self, '_multi_gpu_explicitly_set'):
                self.multi_gpu = True

            # 只在batch_size过大时给出建议，不强制修改
            if self.batch_size > 8:
                print(f"[Config] Warning: DDP mode with batch_size={self.batch_size} may cause memory issues. Consider using --batch_size 8 or smaller.")

            # 只在num_workers过大时给出建议
            if self.num_workers > 2:
                print(f"[Config] Warning: DDP mode with num_workers={self.num_workers} may cause issues. Consider using --num_workers 2 or smaller.")

            self.memory_cleanup_interval = 50
            print(f"[Config] DDP mode detected: batch_size={self.batch_size}, num_workers={self.num_workers}")
        else:
            # 单GPU模式
            if not hasattr(self, '_multi_gpu_explicitly_set'):
                self.multi_gpu = False
            self.memory_cleanup_interval = 100
            print(f"[Config] Single GPU mode: batch_size={self.batch_size}, num_workers={self.num_workers}")

        # 添加内存管理相关配置
        self.save_debug_info = not is_ddp  # DDP模式下关闭调试信息保存

    def _create_directories_safely(self):
        """安全创建目录，避免DDP竞争条件"""
        directories = {
            'log_dir': self.log_dir,
            'pred_dir': self.pred_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'main_model_dir': self.main_model_dir,
            'explanation_generator_dir': self.explanation_generator_dir,
            'vis_dir': self.vis_dir,
            'records_dir': self.records_dir,
            'ID_model_records': self.ID_model_records,
            'explanation_generator_records': self.explanation_generator_records,
            'weights_records': self.weights_records,
        }

        # 只有rank 0创建目录
        if not dist.is_initialized() or dist.get_rank() == 0:
            for name, path in directories.items():
                try:
                    os.makedirs(path, exist_ok=True)
                    print(f"[Rank 0] Created directory: {path}")
                except Exception as e:
                    print(f"[Rank 0] Failed to create {name} at {path}: {e}")

        # 等待rank 0完成目录创建
        if dist.is_initialized():
            dist.barrier()

        # 验证目录是否存在（所有进程）
        for name, path in directories.items():
            if not os.path.exists(path):
                rank = dist.get_rank() if dist.is_initialized() else 0
                raise RuntimeError(f"[Rank {rank}] Directory {path} was not created successfully")

        # 标记目录已创建
        self._directories_created = True

    def ensure_directories_created(self):
        """确保目录已创建，如果没有则创建"""
        if not getattr(self, '_directories_created', False):
            # 首次调用时生成时间戳和路径
            if self.timestamp is None:
                self.timestamp = self._get_unified_timestamp()
                self.run_dir = os.path.join(self.output_dir, self.timestamp)
                self.log_dir = os.path.join(self.run_dir, 'logs')
                self.pred_dir = os.path.join(self.log_dir, 'preds_logs')
                self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
                self.main_model_dir = os.path.join(self.checkpoint_dir, 'main_model')
                self.explanation_generator_dir = os.path.join(self.checkpoint_dir, 'explanation_generator')
                self.vis_dir = os.path.join(self.run_dir, 'visualizations')
                self.records_dir = os.path.join(self.run_dir, 'records')
                self.ID_model_records = os.path.join(self.records_dir, 'ID_model')
                self.explanation_generator_records = os.path.join(self.records_dir, 'explanation_generator')
                self.weights_records = os.path.join(self.records_dir, 'weights')

            self._create_directories_safely()