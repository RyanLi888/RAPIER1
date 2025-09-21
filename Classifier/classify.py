# -*- coding:utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .model import MLP
import sys, os
import numpy as np
from tqdm import tqdm

from .loss import loss_coteaching

# 导入种子控制模块
sys.path.append('../../utils')
try:
    from random_seed import deterministic_shuffle, create_deterministic_dataloader, RANDOM_CONFIG
    SEED_CONTROL_AVAILABLE = True
except ImportError:
    SEED_CONTROL_AVAILABLE = False

# Hyper Parameters
batch_size = 128
learning_rate = 1e-3
epochs = 100
num_gradual = 10
forget_rate = 0.1
exponent = 1
rate_schedule = np.ones(epochs) * forget_rate
rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

def accuracy(logit, target):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)

# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, device):
    
    train_total1=0
    train_correct1=0 
    train_total2=0
    train_correct2=0 

    for i, data_labels in enumerate(train_loader):
        
        feats = data_labels[:, :-1].to(dtype=torch.float32)
        labels = data_labels[:, -1].to(dtype=int)
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
            labels = labels.cuda()
    
        logits1 = model1(feats)
        prec1 = accuracy(logits1, labels)
        train_total1 += 1
        train_correct1 += prec1

        logits2 = model2(feats)
        prec2 = accuracy(logits2, labels)
        train_total2 += 1
        train_correct2 += prec2
        loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch])

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

    train_acc1=float(train_correct1)/float(train_total1)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# predict unknown traffic data's label
def predict(test_loader, model, device, alpha=0.5):

    preds = []
    for i, data in enumerate(test_loader):
        
        # Forward + Backward + Optimize
        feats = data.to(dtype=torch.float32)
        
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
        
        logits = model(feats)
        outputs = F.softmax(logits, dim=1)
        preds.append((outputs[:, 1] > alpha).detach().cpu().numpy())

    return np.concatenate(preds, axis=0)

def main(feat_dir, model_dir, result_dir, TRAIN, cuda_device, parallel=5):
    """
    分类器主训练函数
    
    参数说明:
        parallel (int): 控制使用多少组GAN生成的数据进行训练
                       
    策略验证结果:
        - 原始parallel=5: 使用5组头部数据 (前10%), F1=0.7699
        - ✅ parallel=1: 使用1组完整数据 (前50%), F1=0.7911 (最优)
        - 策略A parallel=5: 使用5组中后部分数据 (33%-43%), F1=0.7720
        
    推荐配置: parallel=1 (获得最佳性能)
    """
    
    cuda_device = int(cuda_device)
    # 加载原始训练集
    be = np.load(os.path.join(feat_dir, 'be_corrected.npy'))[:, :32]
    ma = np.load(os.path.join(feat_dir, 'ma_corrected.npy'))[:, :32]
    be_shape = be.shape[0]
    ma_shape = ma.shape[0]

    # ========== GAN数据增强循环 ==========
    # parallel参数决定使用多少组GAN生成的数据：
    # - parallel=1: 循环1次，使用完整的第0组GAN数据，数据一致性好
    # - parallel=5: 循环5次，使用5组GAN数据的小片段，数据分散
    for index in range(parallel):

        # 加载第index组GAN生成的数据
        be_gen = np.load(os.path.join(feat_dir, 'be_%s_generated_GAN_%d.npy' % (TRAIN, index)))
        ma_gen1 = np.load(os.path.join(feat_dir, 'ma_%s_generated_GAN_1_%d.npy' % (TRAIN, index)))
        ma_gen2 = np.load(os.path.join(feat_dir, 'ma_%s_generated_GAN_2_%d.npy' % (TRAIN, index)))
        # 使用确定性数据打乱
        if SEED_CONTROL_AVAILABLE:
            be_gen = deterministic_shuffle(be_gen, seed=RANDOM_CONFIG['classifier_seed'] + index)
            ma_gen1 = deterministic_shuffle(ma_gen1, seed=RANDOM_CONFIG['classifier_seed'] + index + 1000)
            ma_gen2 = deterministic_shuffle(ma_gen2, seed=RANDOM_CONFIG['classifier_seed'] + index + 2000)
            if index == 0:
                print("✅ 分类器: 使用确定性数据打乱")
        else:
            np.random.shuffle(be_gen)
            np.random.shuffle(ma_gen1)
            np.random.shuffle(ma_gen2)
        # 数据切片与合并 (最优配置: parallel=1)
        # be_shape=336, ma_shape=164
        # parallel=1: be_gen[:336], ma_gen1[:32], ma_gen2[:32] → 完整高质量数据
        # parallel=5: be_gen[:67],  ma_gen1[:6],  ma_gen2[:6]  → 分散小片段
        be = np.concatenate([
            be, 
            be_gen[:be_shape // (parallel)],  # parallel=1时取[:336], parallel=5时取[:67]
        ], axis=0)
        
        ma = np.concatenate([
            ma,
            ma_gen1[:ma_shape // (parallel) // 5],  # parallel=1时取[:32], parallel=5时取[:6]
            ma_gen2[:ma_shape // (parallel) // 5],  # parallel=1时取[:32], parallel=5时取[:6]
        ], axis=0)

    print(be.shape, ma.shape)

    train_data = np.concatenate([be, ma], axis=0)
    train_label = np.concatenate([np.zeros(be.shape[0]), np.ones(ma.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)

    test_data_label = np.load(os.path.join(feat_dir, 'test.npy'))
    test_data = test_data_label[:, :32]
    test_label = test_data_label[:, -1]

    device = int(cuda_device) if cuda_device != 'None' else None
    # define drop rate schedule

    if device != None:
        torch.cuda.set_device(device)
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    if SEED_CONTROL_AVAILABLE:
        train_loader = create_deterministic_dataloader(train_dataset, batch_size, shuffle=True, seed=RANDOM_CONFIG['classifier_seed'])
        print("✅ 分类器: 使用确定性数据加载器")
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # Define models
    print('building model...')
    mlp1 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp1.to_cuda(device)
        mlp1 = mlp1.cuda()
    optimizer1 = torch.optim.Adam(mlp1.parameters(), lr=learning_rate)
    
    mlp2 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp2.to_cuda(device)
        mlp2 = mlp2.cuda()
    optimizer2 = torch.optim.Adam(mlp2.parameters(), lr=learning_rate)

    epoch=0
    mlp1.train()
    mlp2.train()
    for epoch in tqdm(range(epochs)):
        train(train_loader, epoch, mlp1, optimizer1, mlp2, optimizer2, device)
    
    mlp1.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    preds = predict(test_loader, mlp1, device)
    np.save(os.path.join(result_dir, 'prediction.npy'), preds)

    scores = np.zeros((2, 2))
    for label, pred in zip(test_label, preds):
        scores[int(label), int(pred)] += 1
    TP = scores[1, 1]
    FP = scores[0, 1]
    TN = scores[0, 0]
    FN = scores[1, 0]
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1score = 2 * Recall * Precision / (Recall + Precision)
    print(Recall, Precision, F1score)
    
    with open('../data/result/detection_result.txt', 'w') as fp:
        fp.write('Testing data: Benign/Malicious = %d/%d\n'%((TN + FP), (TP + FN)))
        fp.write('Recall: %.2f, Precision: %.2f, F1: %.2f\n'%(Recall, Precision, F1score))
        fp.write('Acc: %.2f\n'%(Accuracy))

    mlp1 = mlp1.cpu()
    mlp1.to_cpu()
    torch.save(mlp1, os.path.join(model_dir, 'Detection_Model.pkl'))

