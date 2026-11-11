"""
日志记录器模块
"""
import os
import logging
from datetime import datetime
import json
import shutil
import torch.distributed as dist

class Logger:
    """日志记录器类"""
    
    def __init__(self, config):
        """
        初始化日志记录器

        参数:
            config: 配置对象
        """
        self.log_dir = config.log_dir
        self.logger = logging.getLogger("vip")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.is_main_process = self.rank == 0

        # 配置日志记录器
        self.logger.setLevel(logging.INFO)

        # 创建格式化器（所有进程都需要）
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 只有rank 0创建文件处理器
        if self.is_main_process:
            # 设置日志文件路径
            self.log_file = os.path.join(self.log_dir, 'train.log')

            # 添加文件处理器
            if not self.logger.handlers:
                fh = logging.FileHandler(self.log_file)
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

        # 添加控制台处理器（所有进程都需要）
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        # 记录实验配置
        self.info(f"日志目录: {self.log_dir}")
        
        # 初始化指标记录
        self.metrics = {}
        
        # 指标记录
        self.metrics_file = os.path.join(self.log_dir, 'metrics.json')
        self.metrics_history = {'train': {}, 'val': {}}
        
        # 只有rank 0处理指标文件
        if self.is_main_process:
            # 如果存在旧的指标文件,加载它
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    self.metrics_history = json.load(f)
        
        # 清理旧日志
        self.cleanup_old_logs()
        
    def info(self, message, rank_prefix=True):
        """记录信息级别的日志"""
        if rank_prefix and dist.is_initialized():
            message = f"[Rank {self.rank}] {message}"

        # 控制台输出（所有进程）
        print(message)

        # 文件写入（只有rank 0）
        if self.is_main_process:
            self.logger.info(message)

    def info_rank0_only(self, message):
        """只在rank 0输出和记录"""
        if self.is_main_process:
            print(f"[Main] {message}")
            self.logger.info(message)

    def warning(self, message):
        """记录警告级别的日志"""
        if dist.is_initialized():
            message = f"[Rank {self.rank}] {message}"
        print(f"WARNING: {message}")
        if self.is_main_process:
            self.logger.warning(message)

    def error(self, message):
        """记录错误级别的日志"""
        if dist.is_initialized():
            message = f"[Rank {self.rank}] {message}"
        print(f"ERROR: {message}")
        if self.is_main_process:
            self.logger.error(message)
        
    def debug(self, message):
        """记录调试级别的日志"""
        self.logger.debug(message)
        
    def log_metrics(self, phase, metrics, epoch):
        """
        记录训练/验证指标
        
        参数:
            phase: 阶段 ('train' 或 'val')
            metrics: 指标字典
            epoch: 当前轮次
        """
        # 确保阶段存在
        if phase not in self.metrics_history:
            self.metrics_history[phase] = {}
            
        # 记录每个指标
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history[phase]:
                self.metrics_history[phase][metric_name] = []
            self.metrics_history[phase][metric_name].append(value)
        
        # 只有rank 0保存到文件
        if self.is_main_process:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            
        # 打印当前指标
        metrics_str = ' - '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.info(f'Epoch {epoch+1} ({phase}): {metrics_str}')
        
    def cleanup_old_logs(self, keep_days=7):
        """
        清理旧的日志文件
        
        参数:
            keep_days: 保留的天数
        """
        current_time = datetime.now()
        
        # 遍历日志目录
        for filename in os.listdir(self.log_dir):
            file_path = os.path.join(self.log_dir, filename)
            
            # 获取文件修改时间
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # 如果文件超过保留天数,删除它
            if (current_time - file_time).days > keep_days:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                self.info(f'已清理旧日志: {filename}')
        
    def log(self, message):
        """记录一般日志信息（等同于info级别）"""
        self.info(message) 