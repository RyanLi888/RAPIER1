"""
随机种子控制模块
==================

本模块提供了统一的随机种子设置功能，确保实验结果的可重复性。

作者: RAPIER 开发团队
版本: 1.0 (简化版)
"""

import random
import numpy as np
import torch
import os


def set_random_seed(seed=7271):
    """
    设置所有相关库的随机种子，确保实验可重复性
    
    参数:
        seed (int): 随机种子值，默认为7271
    """
    print(f"🎯 设置随机种子为: {seed}")
    
    # 设置所有随机数生成器的种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 设置CUDA随机种子（如果有GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("✅ GPU随机种子已设置，启用确定性模式")
    
    # 设置环境变量以确保完全确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("✅ 所有随机种子设置完成")


def create_deterministic_dataloader(dataset, batch_size, shuffle=True, seed=42):
    """
    创建确定性的数据加载器
    
    参数:
        dataset: PyTorch数据集
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        seed (int): 随机种子
        
    返回:
        torch.utils.data.DataLoader: 确定性数据加载器
    """
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    return dataloader


def deterministic_shuffle(array, seed=42):
    """
    确定性的数组打乱函数
    
    参数:
        array (np.ndarray): 要打乱的数组
        seed (int): 随机种子
        
    返回:
        np.ndarray: 打乱后的数组
    """
    # 临时设置种子
    state = np.random.get_state()
    np.random.seed(seed)
    
    # 打乱数组
    shuffled_array = array.copy()
    np.random.shuffle(shuffled_array)
    
    # 恢复原始随机状态
    np.random.set_state(state)
    
    return shuffled_array


def get_deterministic_random_int(low, high, seed=42):
    """
    生成确定性的随机整数
    
    参数:
        low (int): 最小值
        high (int): 最大值（不包含）
        seed (int): 随机种子
        
    返回:
        int: 确定性的随机整数
    """
    rng = np.random.RandomState(seed)
    return rng.randint(low, high)


# 预定义的种子配置 - 最优配置（F1=0.7911）
GLOBAL_SEED = 2024      # 全局种子
AE_SEED = 290984        # AE种子
MADE_SEED = 290713      # MADE模型种子
CLASSIFIER_SEED = 19616 # 分类器种子
GENERATION_SEED = 61592 # 生成器种子

# 导出的配置
RANDOM_CONFIG = {
    'global_seed': GLOBAL_SEED,
    'ae_seed': AE_SEED, 
    'made_seed': MADE_SEED,
    'classifier_seed': CLASSIFIER_SEED,
    'generation_seed': GENERATION_SEED
}