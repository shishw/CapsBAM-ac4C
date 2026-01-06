import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix,matthews_corrcoef, f1_score, roc_auc_score
from model import CapsNet
import time
import re


def load_final_model(model_dir, model_name, device):
    model_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    print(f"加载最终模型: {model_path}")
    
    num_caps = 4  
    
    try:
        match = re.search(r'caps(\d+)', model_name)
        if match:
            num_caps = int(match.group(1))
            print(f"从文件名成功解析胶囊数量: {num_caps}")
        else:
            print(f"文件名中未找到caps数字模式，使用默认值: {num_caps}")
            
    except Exception as e:
        print(f"解析胶囊数量时出错: {e}，使用默认值: {num_caps}")
    
    model = CapsNet(Primary_capsule_num=num_caps)
    model.to(device)
    state_dict = torch.load(model_path, map_location=device,weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"最终模型加载成功，胶囊数量: {num_caps}")
    return model

    
def predict_final_model(model, test_loader, device):
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
        
            #output, _, _ = model(batch_x)
            output=model(batch_x)

            classes = torch.sqrt((output ** 2).sum(2))
            probs = torch.nn.functional.softmax(classes, dim=1)
            
            numpy_probs = probs.detach().cpu().numpy()
            all_probs.extend(numpy_probs)
            all_preds.extend(np.argmax(numpy_probs, axis=1))
    
    return np.array(all_probs), np.array(all_preds)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = load_final_model(args.model_dir, args.model_name, device)

    print(f"从 {args.input} 加载测试数据")
    res = 72 
    
    try:
        with open(args.input, 'r') as f:
            lines = f.readlines()
        
        if lines and lines[0].startswith("figure,label"):
            lines = lines[1:]
        
        images = []
        labels = []  
        
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) < 1:
                continue  
                
            figure_str = parts[0]
            
            if len(parts) >= 2:
                try:
                    label = int(parts[1])
                    labels.append(label)
                except:
                    pass  
            
            try:
                float_arr = np.array(figure_str.split(), dtype=np.float32)
                if len(float_arr) != res * res:
                    raise ValueError(f"图像数据长度应为{res*res}，实际为{len(float_arr)}")
                images.append(float_arr.reshape(res, res))
            except Exception as e:
                print(f"处理第{i+1}行时出错: {e}")
                continue
        
        if len(images) == 0:
            raise ValueError("无法解析任何测试数据")
        
        X_test = torch.tensor(np.array(images)).unsqueeze(1).float()
        y_test = None
        
        if len(labels) == len(images):
            y_test = torch.tensor(labels).long()
            print(f"找到 {len(y_test)} 个标签")
        else:
            print("警告：未找到标签列或标签数量不匹配，将跳过性能评估")
        
        print(f"成功加载 {len(X_test)} 个样本")
        
    except Exception as e:
        print(f"加载测试数据失败: {str(e)}")
        exit(1)

    if y_test is None:
        dummy_labels = torch.zeros(len(X_test))
    else:
        dummy_labels = y_test
    
    test_dataset = TensorDataset(X_test, dummy_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("开始使用最终模型预测...")
    start_time = time.time()
    probabilities, predictions = predict_final_model(model, test_loader, device)
    inference_time = time.time() - start_time
    print(f"推理完成，耗时: {inference_time:.2f}秒")
    
    results = []
    for i in range(len(images)):
        result = {
            'figure': ' '.join(map(str, images[i].flatten())),  # 保持原始格式
            'Probability_0': probabilities[i, 0],
            'Probability_1': probabilities[i, 1],
            'Prediction': predictions[i]
        }
        if y_test is not None:
            result['label'] = y_test[i].item()
        results.append(result)
    
    if y_test is not None:
        y_true = y_test.numpy()
        print("\n最终模型性能:")
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    
        metrics = {
            'ACC': (tp + tn) / (tp + fp + fn + tn),  
            'SEN': tp / (tp + fn),                   
            'PRE': tp / (tp + fp),                   
            'SPEC': tn / (tn + fp),                  
            'MCC': matthews_corrcoef(y_true, predictions),  
            'F1': f1_score(y_true, predictions),            
            'AUROC': roc_auc_score(y_true, probabilities[:, 1])  
        }    
    
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)

        print(f"\n混淆矩阵:")
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\n结果已保存到: {args.output}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="胶囊网络最终模型测试")
    parser.add_argument('--input', type=str, default='CGR_test72.txt')
    parser.add_argument('--output', type=str, default='pred.csv')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=269)                                                          
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)