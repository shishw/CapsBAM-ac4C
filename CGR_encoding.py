import numpy as np
import pandas as pd
import argparse
import os

class FCGR:
    def __init__(self, resolution=64):
        self.resolution = resolution
        self.nucleotide_map = {
            'A': (-1, 1),
            'C': (-1, -1),
            'G': (1, -1),
            'U': (1, 1)
        }
        
    def encode(self, sequence):
        x, y = 0.0, 0.0
        
        fcgr_matrix = np.zeros((self.resolution, self.resolution), dtype=int)
        
        for nucleotide in sequence.upper():
            if nucleotide not in self.nucleotide_map:
                continue  
            
            hx, hy = self.nucleotide_map[nucleotide]
            x = 0.5 * (x + hx)
            y = 0.5 * (y + hy)
            
            i = int(((x + 1) / 2) * (self.resolution - 1))
            j = int(((y + 1) / 2) * (self.resolution - 1))
            
            i = np.clip(i, 0, self.resolution - 1)
            j = np.clip(j, 0, self.resolution - 1)
            
            fcgr_matrix[i, j] += 1
            
        return fcgr_matrix.flatten()

def process_csv(input_file, output_file, resolution=64):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    df = pd.read_csv(input_file)
    
    if 'Sequences' not in df.columns:
        raise ValueError("CSV文件中必须包含'Sequences'列")
    
    fcgr_encoder = FCGR(resolution=resolution)
    
    print(f"正在处理 {len(df)} 条序列，生成FCGR编码 (分辨率: {resolution}x{resolution})...")
    
    df['figure'] = df['Sequences'].apply(lambda seq: 
        ' '.join(map(str, (fcgr_encoder.encode(seq) / fcgr_encoder.encode(seq).max()).flatten().tolist()))
    )
    
    if 'Label' in df.columns:
        df.rename(columns={'Label': 'label'}, inplace=True)
    
    df = df[['figure', 'label']]
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='CGR序列编码工具')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出CSV文件路径')
    parser.add_argument('--resolution', type=int, default=64, help='FCGR矩阵分辨率 (默认: 64)')
    
    args = parser.parse_args()
    
    try:
        process_csv(args.input, args.output, args.resolution)
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()