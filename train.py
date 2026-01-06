#最终模型训练
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.utils.data import DataLoader, TensorDataset
from model import CapsNet 
import copy
import os
import time
from datetime import datetime
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser(description='最终模型训练')
parser.add_argument('--cgr_train', type=str, default='CGR_train72.txt')
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, required=True)  
parser.add_argument('--num_capsules', type=int, default=4)  
args = parser.parse_args()

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

USE_CUDA = torch.cuda.is_available()
print(f"使用 {'GPU' if USE_CUDA else 'CPU'} 进行训练")

set_seed(args.seed)
print(f"设置随机种子: {args.seed}")

SAVE_DIR = args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"最终模型将保存到 {SAVE_DIR} 目录")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(SAVE_DIR, f'final_training_log_{timestamp}.txt')

# 写入训练日志
def write_log(message, print_to_console=True):
    """写入日志文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    with open(log_file, 'a') as f:
        f.write(log_message + "\n")
    
    if print_to_console:
        print(log_message)

# 配置参数
batch_size = args.batch_size
n_epochs = args.epochs
num_cap = args.num_capsules
res = 72 

write_log(f"正在加载训练数据: {args.cgr_train}")
try:
    with open(args.cgr_train, 'r') as f:
        lines = f.readlines()
    
    if lines[0].startswith("figure,label"):
        lines = lines[1:]
    
    str_list = []
    labels = []
    
    for line in lines:
        try:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
                
            feature_str = parts[0]           
            label = int(parts[1])
            
            float_arr = np.array(feature_str.split(), dtype=np.float32)
            temp = float_arr.reshape(res, res)
            
            str_list.append(temp)
            labels.append(label)
        except Exception as e:
            write_log(f"处理数据时出错: {e}")
            continue
    
    if len(str_list) == 0:
        raise ValueError("无法解析任何训练数据")
    
    y = np.array(labels)
    X = torch.tensor(str_list)
    X = X.unsqueeze(1)
    write_log(f"成功加载 {len(X)} 个样本")
    
except Exception as e:
    write_log(f"数据加载失败: {str(e)}")
    exit(1)

# 使用全部数据集训练
dataset = TensorDataset(X, torch.tensor(y))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建最终模型
write_log(f"\n{'='*60}")
write_log(f"开始训练最终模型")
write_log(f"胶囊数量: {num_cap}")
write_log(f"训练轮数: {n_epochs}")
write_log(f"批次大小: {batch_size}")
write_log(f"总样本数: {len(X)}")
write_log(f"正样本: {np.sum(y == 1)}, 负样本: {np.sum(y == 0)}")
write_log(f"{'='*60}")

# 初始化模型
capsule_net = CapsNet(Primary_capsule_num=num_cap)
if USE_CUDA:
    capsule_net = capsule_net.cuda()

optimizer = Adam(capsule_net.parameters(), lr=1e-3, betas=(0.9, 0.999))

# 开始训练
training_start_time = time.time()
train_losses = []
saved_models = []  

for epoch in range(n_epochs):
    epoch_start_time = time.time()
    capsule_net.train()
    
    train_loss = 0
    epoch_predictions = []
    epoch_true_labels = []
    
    for batch_id, (data, target) in enumerate(train_loader):
        epoch_true_labels.extend(target.numpy())
        
        target_onehot = torch.sparse.torch.eye(2).index_select(dim=0, index=target.long())
        data, target_onehot = Variable(data), Variable(target_onehot)

        if USE_CUDA:
            data, target_onehot = data.cuda(), target_onehot.cuda()

        optimizer.zero_grad()
        #output, reconstructions, masked = capsule_net(data)
        output=capsule_net(data)
        #loss = capsule_net.loss(data, output, target_onehot, reconstructions)
        loss=capsule_net.loss(data,output,target_onehot)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        with torch.no_grad():
            v_c = torch.sqrt((output**2).sum(dim=2))
            pred_probs = F.softmax(v_c, dim=1)
            pred_labels = torch.argmax(pred_probs, dim=1)
            epoch_predictions.extend(pred_labels.cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    train_acc = accuracy_score(epoch_true_labels, epoch_predictions)
    train_mcc = matthews_corrcoef(epoch_true_labels, epoch_predictions)
    train_f1 = f1_score(epoch_true_labels, epoch_predictions)
    
    train_losses.append(avg_train_loss)
    epoch_time = time.time() - epoch_start_time
    
    write_log(f"Epoch {epoch+1:2d}/{n_epochs} - "
              f"Loss: {avg_train_loss:.4f} - "
              f"Acc: {train_acc:.4f} - "
              f"MCC: {train_mcc:.4f} - "
              f"F1: {train_f1:.4f} - "
              f"Time: {epoch_time:.2f}s")


    if epoch >= n_epochs - 1:
        model_name = f"caps{num_cap}_epoch{epoch+1}.pth"
        model_path = os.path.join(SAVE_DIR, model_name)

        torch.save(capsule_net.state_dict(), model_path)

        model_info = {
            'epoch': epoch + 1,
            'model_name': model_name,
            'model_path': model_path,
            'accuracy': train_acc,
            'mcc': train_mcc,
            'f1': train_f1,
            'loss': avg_train_loss
        }
        saved_models.append(model_info)

        write_log(f"已保存最终模型: {model_name}")

print(f"\n最终模型训练完成!")
print(f"保存了最后一轮模型到: {SAVE_DIR}")
print(f"训练日志: {log_file}")