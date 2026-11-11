"""
视频重要人物检测系统主入口脚本
"""
import os
import sys
import warnings
import logging
import torch
import time
import signal

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加父目录到Python路径，确保能够导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)


# 绝对导入方式
from npz_llm_model.configs.config import Config
from npz_llm_model.data_processing.data_loader import DataLoader
from npz_llm_model.models.enhanced_transformer_model import EnhancedTransformerModel
from npz_llm_model.train.trainer import Trainer
from npz_llm_model.train.evaluator import Evaluator
from npz_llm_model.train.predictor import Predictor
from npz_llm_model.utils.log_block import log_block

log_block()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置环境变量，减少TensorFlow/absl等冗余输出
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 确保设备顺序一致

def check_cuda_environment():
    """检查CUDA环境"""
    try:
        logger.info("=== CUDA环境检查 ===")
        logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            logger.info(f"当前GPU: {torch.cuda.current_device()}")
            logger.info(f"GPU名称: {torch.cuda.get_device_name()}")
        logger.info("==================")
        return True
    except Exception as e:
        logger.error(f"CUDA环境检查失败: {e}")
        return False

def safe_cuda_init():
    """安全初始化CUDA"""
    max_retries = 3
    for i in range(max_retries):
        try:
            if not torch.cuda.is_available():
                logger.error("CUDA不可用")
                return False
                
            n_gpus = torch.cuda.device_count()
            logger.info(f"检测到{n_gpus}个GPU")
            
            # 初始化CUDA
            os.system("conda activate vip")
            torch.cuda.init()
            logger.info("CUDA初始化成功")
            
            return True
        except Exception as e:
            logger.error(f"CUDA初始化失败 (尝试 {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(1)
    return False

def init_distributed(config):
    """初始化分布式训练环境"""
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，无法进行分布式训练")
            
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # 设置当前设备
        torch.cuda.set_device(local_rank)
        
        # 初始化进程组
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
        
        config.device = f'cuda:{local_rank}'
        logger.info(f"分布式训练初始化成功: rank={local_rank}, world_size={world_size}")
        return True
    except Exception as e:
        logger.error(f"分布式训练初始化失败: {e}")
        return False

def kill_children(signum, frame):
    print("\n收到中断信号，正在杀死所有子进程...")
    try:
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    except Exception as e:
        print(f"进程组杀死失败: {e}")
    sys.exit(0)

def main():
    """主函数"""
    # 解析命令行参数
    args = Config.get_args()
    
    # 初始化配置
    config = Config()
    config.update_from_args(args)

    # 根据训练环境自动调整配置
    config.adapt_config_for_training_mode()
    
    # 设置测试模式
    if args.test:
        config.test_mode = True
        logger.info("⚠️ 测试模式已启用")
    
    # 检查CUDA环境
    if not check_cuda_environment():
        logger.error("CUDA环境检查失败，退出程序")
        return
    
    # 安全初始化CUDA
    if not safe_cuda_init():
        logger.error("CUDA初始化失败，退出程序")
        return
    
    # 自动检测多GPU环境
    if torch.cuda.device_count() > 1 and 'WORLD_SIZE' in os.environ:
        config.multi_gpu = True

    # 多GPU训练设置
    if config.multi_gpu and torch.cuda.device_count() > 1:
        if not init_distributed(config):
            logger.error("分布式训练初始化失败，退出程序")
            return

    # 分布式初始化完成后，确保目录创建
    config.ensure_directories_created()

    try:
        # 创建数据加载器
        data_loader = DataLoader(config)
        
        # 根据运行模式执行相应操作
        if config.mode == 'train':
            # 创建模型
            logger.info("使用增强型Transformer模型（多粒度时空感知融合）")
            model = EnhancedTransformerModel(config)
            
            # 创建训练器和评估器
            trainer = Trainer(model, data_loader, config)
            evaluator = Evaluator(model, data_loader, config)
            
            # 如果指定了模型路径，加载检查点
            if config.model_path and os.path.exists(config.model_path):
                trainer.load_checkpoint(config.model_path)
            
            # 训练模型
            trainer.train(evaluator)
        
        elif config.mode == 'predict':
            # 创建模型
            logger.info("使用增强型Transformer模型（多粒度时空感知融合）")
            model = EnhancedTransformerModel(config)
            
            # 创建预测器
            if config.model_path:
                logger.info(f"将使用指定的模型权重: {config.model_path}")
            else:
                logger.info("将使用当前训练的最佳模型权重")
            predictor = Predictor(model, data_loader, config)
            
            # 在测试集上进行预测
            logger.info("开始预测...")
            import numpy as np
            repeats = getattr(config, 'predict_repeats', 1)
            split = getattr(config, 'predict_split', 'val')
            total_samples = 0
            repeat_metrics = []
            per_run_rows = []
            for epoch_idx in range(repeats):
                logger.info(f"预测循环 {epoch_idx+1}/{repeats} - 使用split={split}")
                # 默认行为：在每次重复前随机化 flow_cnn（用于估计随机初始化带来的方差）
                base_seed = int(time.time())
                seed = base_seed + epoch_idx
                try:
                    ok = predictor.randomize_flow_cnn(seed=seed)
                    logger.info(f"在重复 {epoch_idx} 前随机化 flow_cnn: 成功={ok}, seed={seed}")
                except Exception as e:
                    logger.error(f"随机化 flow_cnn 失败: {e}")
                predictions = predictor.predict(split, epoch=epoch_idx)
                n = len(predictions)
                total_samples += n
                # Safely aggregate rank1/2/3 from predictions
                if n > 0:
                    r1 = sum(int(p.get('rank1_correct', p.get('correct', 0))) for p in predictions) / n
                    r2 = sum(int(p.get('rank2_correct', p.get('rank1_correct', p.get('correct', 0)))) for p in predictions) / n
                    r3 = sum(int(p.get('rank3_correct', p.get('rank1_correct', p.get('correct', 0)))) for p in predictions) / n
                else:
                    r1 = r2 = r3 = 0.0
                logger.info(f"本次预测完成: 处理 {n} 个样本, Rank-1: {r1:.4f}, Rank-2: {r2:.4f}, Rank-3: {r3:.4f}")
                repeat_metrics.append({'rank1': r1, 'rank2': r2, 'rank3': r3, 'n': n})
                per_run_rows.append({'run_idx': epoch_idx, 'rank1': r1, 'rank2': r2, 'rank3': r3, 'samples': n})

            # 计算 mean/std
            if repeat_metrics:
                rank1s = np.array([m['rank1'] for m in repeat_metrics])
                rank2s = np.array([m['rank2'] for m in repeat_metrics])
                rank3s = np.array([m['rank3'] for m in repeat_metrics])
                summary = {
                    'rank1_mean': float(rank1s.mean()), 'rank1_std': float(rank1s.std()),
                    'rank2_mean': float(rank2s.mean()), 'rank2_std': float(rank2s.std()),
                    'rank3_mean': float(rank3s.mean()), 'rank3_std': float(rank3s.std()),
                    'repeats': repeats,
                    'total_samples': total_samples
                }
                logger.info(f"重复预测统计结果 (mean ± std): Rank-1: {summary['rank1_mean']:.4f} ± {summary['rank1_std']:.4f}, Rank-2: {summary['rank2_mean']:.4f} ± {summary['rank2_std']:.4f}, Rank-3: {summary['rank3_mean']:.4f} ± {summary['rank3_std']:.4f}")
                # 保存到 records 目录
                try:
                    import csv
                    agg_path = os.path.join(config.ID_model_records, 'predict_repeats_summary.csv')
                    with open(agg_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=['metric', 'mean', 'std'])
                        writer.writeheader()
                        writer.writerow({'metric': 'rank1', 'mean': f"{summary['rank1_mean']:.6f}", 'std': f"{summary['rank1_std']:.6f}"})
                        writer.writerow({'metric': 'rank2', 'mean': f"{summary['rank2_mean']:.6f}", 'std': f"{summary['rank2_std']:.6f}"})
                        writer.writerow({'metric': 'rank3', 'mean': f"{summary['rank3_mean']:.6f}", 'std': f"{summary['rank3_std']:.6f}"})
                    logger.info(f"重复预测摘要已保存至: {agg_path}")
                    # 保存每次的行
                    per_run_path = os.path.join(config.ID_model_records, 'predict_repeats_per_run.csv')
                    with open(per_run_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=['run_idx', 'rank1', 'rank2', 'rank3', 'samples'])
                        writer.writeheader()
                        for row in per_run_rows:
                            writer.writerow(row)
                    logger.info(f"每次重复结果已保存至: {per_run_path}")
                except Exception as e:
                    logger.error(f"保存重复预测统计失败: {e}")
            else:
                logger.info("没有可用的重复预测结果")

            # End predict branch

    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        raise e

if __name__ == "__main__":
    # 捕获SIGINT信号（如Ctrl+C），优雅地终止所有子进程
    signal.signal(signal.SIGINT, kill_children)
    
    main()