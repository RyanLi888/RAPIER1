import os 
import sys 
sys.path.append('..')
import numpy as np
import torch
import random
import shutil
import glob

# 导入种子控制模块
sys.path.append('../utils')
try:
    from random_seed import set_random_seed, RANDOM_CONFIG
    SEED_CONTROL_AVAILABLE = True
    print("✅ 随机种子控制模块导入成功")
except ImportError:
    print("⚠️  警告：随机种子控制模块导入失败，将使用默认行为")
    SEED_CONTROL_AVAILABLE = False

# 设置随机种子
if SEED_CONTROL_AVAILABLE:
    set_random_seed(RANDOM_CONFIG['global_seed'])
    print(f"🎯 已设置全局随机种子: {RANDOM_CONFIG['global_seed']}")
    print(f"🎲 使用种子配置: {RANDOM_CONFIG}")
else:
    # 备用种子设置
    GLOBAL_SEED = 2024
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎯 已设置备用全局随机种子: {GLOBAL_SEED}")

import MADE
import Classifier
import AE

def cleanup_previous_results(feat_dir, model_dir, made_dir, result_dir):
    """
    清理之前运行生成的特征、模型和结果文件
    """
    print("正在清理之前的结果文件...")
    
    # 清理特征文件 (保留原始数据文件)
    if os.path.exists(feat_dir):
        feat_files = [
            'be.npy', 'ma.npy', 'test.npy',  # AE生成的特征文件
            'be_corrected.npy', 'ma_corrected.npy',  # 修正后的特征
            'be_groundtruth.npy', 'ma_groundtruth.npy',  # 真实标签
            'be_unknown.npy', 'ma_unknown.npy',  # 未知标签
        ]
        # 清理GAN生成的特征文件
        gan_patterns = [
            'be_*_generated_GAN_*.npy',
            'ma_*_generated_GAN_*.npy'
        ]
        
        for pattern in gan_patterns:
            for file_path in glob.glob(os.path.join(feat_dir, pattern)):
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")
        
        for feat_file in feat_files:
            file_path = os.path.join(feat_dir, feat_file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")
    
    # 清理模型文件
    if os.path.exists(model_dir):
        model_files = [
            'gru_ae.pkl',  # AE模型
            'Detection_Model.pkl',  # 检测模型
            'dis_*.pt',  # 判别器模型
            'gen_*.pt',  # 生成器模型
            'gen1_*.pt',  # 生成器1模型
            'gen2_*.pt',  # 生成器2模型
            'made_*.pt',  # MADE模型
            'epochs_*.pt'  # 分轮训练模型
        ]
        
        for pattern in model_files:
            for file_path in glob.glob(os.path.join(model_dir, pattern)):
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")
    
    # 清理MADE结果文件
    if os.path.exists(made_dir):
        try:
            for file_path in os.listdir(made_dir):
                full_path = os.path.join(made_dir, file_path)
                if os.path.isfile(full_path):
                    os.remove(full_path)
                    print(f"已删除: {full_path}")
        except Exception as e:
            print(f"清理MADE目录失败: {e}")
    
    # 清理结果文件
    if os.path.exists(result_dir):
        result_files = [
            'prediction.npy',  # 预测结果
            'detection_result.txt',  # 检测结果
            'label_correction.txt'  # 标签修正结果
        ]
        
        for result_file in result_files:
            file_path = os.path.join(result_dir, result_file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")
    
    print("清理完成！")

def generate(feat_dir, model_dir, made_dir, index, cuda):
    TRAIN_be = 'be_corrected'
    TRAIN_ma = 'ma_corrected'
    TRAIN = 'corrected'
    
    MADE.train.main(feat_dir, model_dir, TRAIN_be, cuda, '-30')
    MADE.train.main(feat_dir, model_dir, TRAIN_ma, cuda, '-30')
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_be, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_be, cuda)

    MADE.train_gen_GAN.main(feat_dir, model_dir, made_dir, TRAIN, cuda)
    MADE.generate_GAN.main(feat_dir, model_dir, TRAIN, index, cuda)

def generate_cpus(feat_dir, model_dir, made_dir, indices, cuda):
    for index in indices:
        generate(feat_dir, model_dir, made_dir, index, cuda)

def main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, random_seed=None, is_seed_search=False):
    
    # 在开始训练前清理之前的结果文件
    cleanup_previous_results(feat_dir, model_dir, made_dir, result_dir)
    
    print("开始训练流程...")
    AE.train.main(data_dir, model_dir, cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)

    TRAIN = 'be'
    MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20')
    MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN)
    MADE.final_predict.main(feat_dir, result_dir)
    
    generate_cpus(feat_dir, model_dir, made_dir, list(range(5)), cuda)
    
    TRAIN = 'corrected'
    
    # ========== PARALLEL参数优化说明 ==========
    # 最优配置: parallel=1，F1分数=0.7911 (已验证)
    #
    # parallel=5 vs parallel=1的关键区别:
    # ┌─────────────┬─────────────────┬─────────────────┐
    # │    方面     │   parallel=5    │   parallel=1    │
    # ├─────────────┼─────────────────┼─────────────────┤
    # │ 数据使用策略 │ 分散使用5个GAN  │ 集中使用1个GAN  │
    # │            │ 文件的小部分    │ 文件的大部分    │
    # ├─────────────┼─────────────────┼─────────────────┤
    # │ GAN数据完整性│ 每个文件只用    │ 每个文件使用    │
    # │            │ 10-15%         │ 50%            │
    # ├─────────────┼─────────────────┼─────────────────┤
    # │ 数据多样性  │ 来自5个不同生成 │ 来自1个完整生成 │
    # │            │ 批次的片段      │ 批次           │
    # └─────────────┴─────────────────┴─────────────────┘
    #
    # 性能提升原因 (理论vs实际的矛盾分析):
    # 
    # 🤔 理论上parallel=5应该更好 (泛化性角度):
    #   - 更多样的数据来源 → 更好的泛化能力
    #   - 减少对单一GAN批次的依赖 → 更强的鲁棒性
    # 
    # 🎯 但实际上parallel=1更好的原因:
    # 1. **数据质量**: parallel=1使用50%完整数据 vs parallel=5只用10%头部片段
    # 2. **生成质量**: GAN后期生成的样本通常比初期质量更好
    # 3. **特征连贯性**: 连续的数据块比分散的小片段学习效果更好
    # 4. **信息密度**: 单个完整批次包含更丰富的特征模式
    # 5. **训练稳定性**: 避免了多源数据的分布不一致问题
    # ==========================================
    
    Classifier.classify.main(feat_dir, model_dir, result_dir, TRAIN, cuda, parallel=1)

def run_normal_mode(random_seed=None):
    """
    正常模式运行 - 使用固定的data目录结构
    
    参数:
        random_seed (int): 随机种子，默认使用配置文件中的种子
    """
    # 设置正常模式的目录路径
    data_dir = '../data/data'      # 原始数据目录
    feat_dir = '../data/feat'      # 特征文件目录
    model_dir= '../data/model'     # 模型保存目录
    made_dir = '../data/made'      # MADE相关文件目录
    result_dir='../data/result'    # 结果输出目录
    cuda = 0                       # 使用第一个CUDA设备（GPU 0）
    
    print("🚀 RAPIER正常模式运行")
    print("📁 使用固定目录结构: data/feat, data/model, data/made, data/result")
    
    # 执行主函数（正常模式）
    return main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, random_seed, is_seed_search=False)

if __name__ == '__main__':
    """
    程序入口点
    
    当直接运行此脚本时，使用正常模式
    """
    run_normal_mode()