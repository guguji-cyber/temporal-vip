"""
基础模型类模块
"""
import torch
import torch.nn as nn
import os


class BaseModel(nn.Module):
    """视频重要人物检测基础模型类"""
    
    def __init__(self, config):
        """
        初始化基础模型
        
        参数:
            config: 配置对象
        """
        super(BaseModel, self).__init__()
        self.config = config
    
    def forward(self, batch_dict):
        """
        前向传播
        
        参数:
            batch_dict: 批次数据字典
        
        返回:
            output_dict: 输出字典
        """
        raise NotImplementedError("基类的前向传播方法需要被子类实现")
    
    def save_checkpoint(self, epoch, optimizer=None, scheduler=None, metrics=None, extra_info=None, is_best=False):
        """
        专业保存模型检查点，支持完整训练状态，兼容主流程
        """
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': self._serialize_config(self.config),
            'extra_info': extra_info or {}
        }
        epoch_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"已保存最佳模型到 {best_path}")
        print(f"已保存第{epoch}轮检查点到 {epoch_path}")

    @staticmethod
    def _serialize_config(config):
        """智能序列化config，兼容各种类型"""
        if hasattr(config, 'to_dict'):
            return config.to_dict()
        elif hasattr(config, '__dict__'):
            return dict(config.__dict__)
        else:
            return str(config)

    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None, strict=False, load_optimizer=True, load_scheduler=True):
        """
        专业加载模型检查点，支持严格/兼容模式，详细log所有不匹配参数，兼容主流程
        """
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # 加载模型状态
        try:
            missing, unexpected = self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            if not strict:
                print(f"[兼容模式] 未加载参数: {missing}")
                print(f"[兼容模式] 多余参数: {unexpected}")
        except Exception as e:
            print(f"模型权重加载异常: {e}")
        # 加载优化器
        if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"优化器参数加载失败: {e}")
        # 加载调度器
        if scheduler is not None and load_scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"调度器参数加载失败: {e}")
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', None)
        print(f"已加载模型，当前epoch: {epoch}")
        return epoch, metrics 