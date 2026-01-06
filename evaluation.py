# 4折交叉验证
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from model import CapsNet  
import copy
import os
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser(description='4折交叉验证')
parser.add_argument('--cgr_train', type=str, default='CGR_train72.txt')
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=75)
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
print(f"模型和日志将保存到 {SAVE_DIR} 目录")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_log_file = os.path.join(SAVE_DIR, f'training_log_{timestamp}.txt')

def init_logging():
    with open(main_log_file, 'w') as f:
        f.write(f"Training Log - Started at {datetime.now()}\n")
        f.write(f"Arguments: {vars(args)}\n")
        f.write(f"Device: {'GPU' if USE_CUDA else 'CPU'}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Total samples: {len(X) if 'X' in globals() else 'Unknown'}\n")
        f.write("="*80 + "\n\n")

def write_log(message, print_to_console=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"

    with open(main_log_file, 'a') as f:
        f.write(log_message + "\n")

    if print_to_console:
        print(log_message)

# 配置参数
batch_size = args.batch_size
n_epochs = args.epochs
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


init_logging()

dataset = TensorDataset(X, torch.tensor(y))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_cap_list = [4]

kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)

for num_cap in num_cap_list:
    epoch_accuracies = []  
    epoch_models = []
    epoch_metrics = {'mcc': [], 'f1': [], 'auc': [], 'sen': [], 'spe': []}  
    write_log(f"开始训练胶囊数={num_cap}的配置，将跟踪每个epoch的平均准确率")

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        fold_start_time = time.time()

        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        capsule_net = CapsNet(Primary_capsule_num=num_cap)
        if USE_CUDA:
            capsule_net = capsule_net.cuda()
        optimizer = Adam(capsule_net.parameters(), lr=1e-3, betas=(0.9, 0.999))
        
        write_log(f"\n{'='*50}")
        write_log(f"训练开始: Fold {fold+1}/4, Capsules={num_cap}")
        write_log(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
        write_log(f"批次大小: {batch_size}, 训练轮数: {n_epochs}")
        write_log(f"{'='*50}")

        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            capsule_net.train()
            train_loss = 0

            for batch_id, (data, target) in enumerate(train_loader):
                target = torch.sparse.torch.eye(2).index_select(dim=0, index=target.long())
                data, target = Variable(data), Variable(target)

                if USE_CUDA:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                #output, reconstructions, masked = capsule_net(data)
                output = capsule_net(data)
                #loss = capsule_net.loss(data, output, target, reconstructions)
                loss = capsule_net.loss(data, output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            epoch_time = time.time() - epoch_start_time

            write_log(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Time: {epoch_time:.2f}s")

            capsule_net.eval()
            with torch.no_grad():
                all_true_labels = []
                all_pred_labels = []
                all_pred_probs = []

                for data, target in val_loader:
                    true_labels = target.numpy()
                    target_onehot = torch.sparse.torch.eye(2).index_select(dim=0, index=target.long())

                    if USE_CUDA:
                        data, target_onehot = data.cuda(), target_onehot.cuda()

                    #output, reconstructions, masked = capsule_net(data)
                    output = capsule_net(data)

                    v_c = torch.sqrt((output**2).sum(dim=2))
                    pred_probs = F.softmax(v_c, dim=1)
                    pred_labels = torch.argmax(pred_probs, dim=1)

                    all_true_labels.extend(true_labels)
                    all_pred_labels.extend(pred_labels.cpu().numpy())
                    all_pred_probs.extend(pred_probs[:, 1].cpu().numpy())

                acc = accuracy_score(all_true_labels, all_pred_labels)
                mcc = matthews_corrcoef(all_true_labels, all_pred_labels)
                f1 = f1_score(all_true_labels, all_pred_labels)
                auc_score = roc_auc_score(all_true_labels, all_pred_probs)
                
                tn, fp, fn, tp = confusion_matrix(all_true_labels, all_pred_labels).ravel()
                sen = tp / (tp + fn)  
                spe = tn / (tn + fp)  

                write_log(f"验证结果 - ACC: {acc:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}, Sen: {sen:.4f}, Spe: {spe:.4f}")

                if fold == 0:  
                    epoch_accuracies.append([])
                    epoch_models.append([])
                    for key in epoch_metrics:
                        epoch_metrics[key].append([])

                epoch_accuracies[epoch].append(acc)  
                epoch_models[epoch].append(copy.deepcopy(capsule_net.state_dict()))
                epoch_metrics['mcc'][epoch].append(mcc)
                epoch_metrics['f1'][epoch].append(f1)
                epoch_metrics['auc'][epoch].append(auc_score)
                epoch_metrics['sen'][epoch].append(sen)
                epoch_metrics['spe'][epoch].append(spe)


        fold_time = time.time() - fold_start_time
        write_log(f"Fold {fold+1} 完成，耗时: {fold_time:.2f}s")

    avg_accuracies = [np.mean(epoch_acc) for epoch_acc in epoch_accuracies]
    best_avg_epoch = np.argmax(avg_accuracies)
    best_avg_acc = avg_accuracies[best_avg_epoch]
    
    best_avg_mcc = np.mean(epoch_metrics['mcc'][best_avg_epoch])
    best_avg_f1 = np.mean(epoch_metrics['f1'][best_avg_epoch])
    best_avg_auc = np.mean(epoch_metrics['auc'][best_avg_epoch])
    best_avg_sen = np.mean(epoch_metrics['sen'][best_avg_epoch])
    best_avg_spe = np.mean(epoch_metrics['spe'][best_avg_epoch])

    write_log(f"\n{'='*50}")
    write_log(f"所有折训练完成!")
    write_log(f"最佳平均epoch: {best_avg_epoch+1}")
    write_log(f"平均指标 - ACC: {best_avg_acc:.4f}, MCC: {best_avg_mcc:.4f}, F1: {best_avg_f1:.4f}, AUC: {best_avg_auc:.4f}, Sen: {best_avg_sen:.4f}, Spe: {best_avg_spe:.4f}")
    write_log(f"{'='*50}")


    for fold_idx in range(4):

        model_state_to_save = epoch_models[best_avg_epoch][fold_idx]
        
        model_name = f"fold{fold_idx+1}_epoch{best_avg_epoch+1}_acc{best_avg_acc:.4f}.pth"
        save_path = os.path.join(SAVE_DIR, model_name)
        
        torch.save(model_state_to_save, save_path)
        write_log(f"最佳平均epoch模型 (Fold {fold_idx+1}) 已保存: {save_path}")


write_log(f"\n训练完成! 所有模型和日志保存在: {SAVE_DIR}")
write_log(f"日志文件: {main_log_file}")

print(f"\n训练完成!")
print(f"模型和日志目录: {SAVE_DIR}")
print(f"日志文件: {main_log_file}")