import os
import re
import argparse
import matplotlib.pyplot as plt

def parse_results(file_path):
    """解析 results.txt 文件并提取所有的 loss 和 metrics"""
    data = {
        'epochs': [],
        'train_loss': [], 'train_traj_loss': [], 'train_bldg_loss': [], 'train_src_loss': [],
        'val_loss': [], 'val_traj_loss': [], 'val_bldg_loss': [], 'val_src_loss': [],
        'train_traj_metrics': {'P': [], 'R': [], 'F1': [], 'IOU': []},
        'train_bldg_metrics': {'P': [], 'R': [], 'F1': [], 'IOU': []},
        'train_src_metrics': {'P': [], 'R': [], 'F1': [], 'IOU': []},
        'val_traj_metrics': {'P': [], 'R': [], 'F1': [], 'IOU': []},
        'val_bldg_metrics': {'P': [], 'R': [], 'F1': [], 'IOU': []},
        'val_src_metrics': {'P': [], 'R': [], 'F1': [], 'IOU': []},
    }

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到日志文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_epoch = None
    phase = 'train'  # 标记当前正在读取的是 Train 还是 Val

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 匹配 Epoch
        m = re.match(r'===== Epoch (\d+) =====', line)
        if m:
            current_epoch = int(m.group(1))
            data['epochs'].append(current_epoch)
            phase = 'train'  # 每个 epoch 先打印 train
            continue

        # 遇到 Val Loss 说明接下来是验证集数据
        if line.startswith('Val Loss:'):
            phase = 'val'

        # 解析 Losses
        if line.startswith('Train Loss:'): data['train_loss'].append(float(line.split(':')[-1]))
        elif line.startswith('Train Traj Loss:'): data['train_traj_loss'].append(float(line.split(':')[-1]))
        elif line.startswith('Train Bldg Loss:'): data['train_bldg_loss'].append(float(line.split(':')[-1]))
        elif line.startswith('Train Src Loss:'): data['train_src_loss'].append(float(line.split(':')[-1]))

        elif line.startswith('Val Loss:'): data['val_loss'].append(float(line.split(':')[-1]))
        elif line.startswith('Val Traj Loss:'): data['val_traj_loss'].append(float(line.split(':')[-1]))
        elif line.startswith('Val Bldg Loss:'): data['val_bldg_loss'].append(float(line.split(':')[-1]))
        elif line.startswith('Val Src Loss:'): data['val_src_loss'].append(float(line.split(':')[-1]))

        # 解析 Metrics (Traj, Bldg, Src)
        elif line.startswith('Traj -') or line.startswith('Bldg -') or line.startswith('Src'):
            # 处理字符串 "Src  -" 或者 "Src -"
            parts = line.split('-')
            prefix = parts[0].strip().lower()  # traj, bldg, src
            metrics_str = parts[1].strip()     # P:xxx R:xxx F1:xxx IOU:xxx
            
            p_val = float(re.search(r'P:([0-9.]+)', metrics_str).group(1))
            r_val = float(re.search(r'R:([0-9.]+)', metrics_str).group(1))
            f1_val = float(re.search(r'F1:([0-9.]+)', metrics_str).group(1))
            iou_val = float(re.search(r'IOU:([0-9.]+)', metrics_str).group(1))
            
            target_dict = data[f'{phase}_{prefix}_metrics']
            target_dict['P'].append(p_val)
            target_dict['R'].append(r_val)
            target_dict['F1'].append(f1_val)
            target_dict['IOU'].append(iou_val)
            
    return data

def plot_curves(data, save_dir):
    """绘制所有曲线并保存"""
    os.makedirs(save_dir, exist_ok=True)
    epochs = data['epochs']

    # ================= 1. 总loss和3个子loss的同一张图 =================
    # 训练集 Losses 集成图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_loss'], label='Total Loss', linewidth=2.5, color='black')
    plt.plot(epochs, data['train_traj_loss'], label='Traj Loss', linestyle='--')
    plt.plot(epochs, data['train_bldg_loss'], label='Bldg Loss', linestyle='--')
    plt.plot(epochs, data['train_src_loss'], label='Src Loss', linestyle='--')
    plt.title('Training Losses Overview')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'all_losses_train.png'), dpi=150)
    plt.close()

    # 验证集 Losses 集成图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['val_loss'], label='Total Loss', linewidth=2.5, color='black')
    plt.plot(epochs, data['val_traj_loss'], label='Traj Loss', linestyle='--')
    plt.plot(epochs, data['val_bldg_loss'], label='Bldg Loss', linestyle='--')
    plt.plot(epochs, data['val_src_loss'], label='Src Loss', linestyle='--')
    plt.title('Validation Losses Overview')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'all_losses_val.png'), dpi=150)
    plt.close()

    # ================= 2. 每个loss的单独图 (Train vs Val 对比) =================
    loss_types = [
        ('loss', 'Total Loss'), 
        ('traj_loss', 'Traj Loss'), 
        ('bldg_loss', 'Building Loss'), 
        ('src_loss', 'Source Loss')
    ]
    for key, name in loss_types:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, data[f'train_{key}'], label='Train', marker='o', markersize=4)
        plt.plot(epochs, data[f'val_{key}'], label='Val', marker='s', markersize=4)
        plt.title(f'{name} (Train vs Val)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(os.path.join(save_dir, f'compare_{key}.png'), dpi=150)
        plt.close()

    # ================= 3. Metrics的各种图 =================
    components = ['traj', 'bldg', 'src']
    metric_names = ['P', 'R', 'F1', 'IOU']
    
    for comp in components:
        # 为每个组件（如Traj）生成一个包含4个子图的面板
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Metrics for {comp.upper()}', fontsize=16)
        
        for i, metric in enumerate(metric_names):
            ax = axs[i//2, i%2]
            ax.plot(epochs, data[f'train_{comp}_metrics'][metric], label='Train', marker='o', markersize=4)
            ax.plot(epochs, data[f'val_{comp}_metrics'][metric], label='Val', marker='s', markersize=4)
            ax.set_title(f'{metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.5)
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'metrics_{comp}.png'), dpi=150)
        plt.close()

    print(f"✅ 所有图表已成功绘制并保存至: {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制网络训练的 Loss 和 Metrics 曲线')
    parser.add_argument('--exp_name', type=str, required=True, help='实验名称 (例如: delvmap_exp1)')
    parser.add_argument('--base_dir', type=str, default='checkpoints', help='checkpoints 所在的父级相对路径')
    
    args = parser.parse_args()

    # 如果你是在根目录下运行，默认路径就是 ./checkpoints/exp_name
    exp_dir = os.path.join(args.base_dir, args.exp_name)
    results_txt_path = os.path.join(exp_dir, 'results.txt')
    vis_log_dir = os.path.join(exp_dir, 'vis_log')

    # 解析并画图
    try:
        parsed_data = parse_results(results_txt_path)
        plot_curves(parsed_data, vis_log_dir)
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        

# python Dual_Signal_Fusion_based_Map_Completion/plot_results.py --exp_name delvmap_exp2
# 保存在 checkpoints/delvmap_exp2/vis_log/ 目录下
