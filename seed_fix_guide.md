# RAPIER1 种子配置修复指南

## 问题分析

RAPIER1和RAPIER-master的种子配置存在差异，导致结果不一致：

- **RAPIER-master**: F1=0.7911 (最优结果)
- **RAPIER1**: F1=0.7025 (不一致)

## 根本原因

1. **种子设置时机不同**:
   - RAPIER-master: 在main函数内部调用`set_random_seed()`
   - RAPIER1: 在main.py开头直接设置全局种子

2. **种子设置方式不同**:
   - RAPIER-master: 使用统一的种子控制模块
   - RAPIER1: 各个模块独立设置种子

## 修复方案

### 方案1: 创建统一的种子控制模块 (推荐)

1. 在RAPIER1中创建`utils/random_seed.py`文件
2. 复制RAPIER-master的种子控制逻辑
3. 修改main.py使用统一的种子设置

### 方案2: 修改现有种子设置 (简单)

修改RAPIER1的main.py，确保种子设置的顺序和方式与RAPIER-master一致。

## 具体修复步骤

### 步骤1: 创建utils目录和种子控制模块

```bash
mkdir -p /home/lx/python/RAPIER1/utils
```

### 步骤2: 复制种子控制模块

从RAPIER-master复制`utils/random_seed.py`到RAPIER1

### 步骤3: 修改main.py

将main.py开头的种子设置代码替换为：

```python
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
```

## 验证方法

修复后运行RAPIER1，应该得到与RAPIER-master一致的结果：
- F1分数: 0.7911
- 准确率: 0.9640
- 召回率: 0.7500
- 精确率: 0.8371

## 注意事项

1. 确保所有模块的种子设置都正确
2. 检查是否有其他环境差异
3. 验证数据文件是否一致
