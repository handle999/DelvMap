import sys
sys.path.append('./')
import torch
import numpy as np

# 加载数据
from Dual_Signal_Fusion_based_Map_Completion.data_loader import get_data_loader_multistage
dl = get_data_loader_multistage('dataset/xian_2019_delvmap/', 'train', batch_size=2)
batch = next(iter(dl))

print("=" * 60)
print("1. 输入数据检查 (修复后)")
print("=" * 60)
print("traj_data: min=%.4f, max=%.4f" % (batch['traj_data'].min(), batch['traj_data'].max()))
print("src_data: min=%.4f, max=%.4f" % (batch['src_data'].min(), batch['src_data'].max()))
print("label_data: min=%.4f, max=%.4f, unique=%s" % (batch['label_data'].min(), batch['label_data'].max(), torch.unique(batch['label_data']).tolist()))
print("building_label_data: min=%.4f, max=%.4f, unique=%s" % (batch['building_label_data'].min(), batch['building_label_data'].max(), torch.unique(batch['building_label_data']).tolist()))

print("\n" + "=" * 60)
print("2. 模型输出和Metrics检查")
print("=" * 60)

# 直接创建模型
from Dual_Signal_Fusion_based_Map_Completion.models.translator import DSFNet
from Dual_Signal_Fusion_based_Map_Completion.models.metrics import Metrics

input_nc = 5
net = DSFNet(input_nc)
metrics = Metrics()

# 前向传播
traj = batch['traj_data']
src = batch['src_data']
label = batch['label_data']

pred_traj, pred_building, src_pred_traj = net(traj, src)

print("\n模型输出:")
print("pred_traj: min=%.4f, max=%.4f" % (pred_traj.min(), pred_traj.max()))
print("pred_building: min=%.4f, max=%.4f" % (pred_building.min(), pred_building.max()))
print("src_pred_traj: min=%.4f, max=%.4f" % (src_pred_traj.min(), src_pred_traj.max()))

print("\nMetrics计算:")
m = metrics(pred_traj, label)
print("traj metrics [P,R,F1,IOU]:", m.numpy())

m2 = metrics(src_pred_traj, label)
print("src_pred_traj metrics [P,R,F1,IOU]:", m2.numpy())

# 测试F1不为0的情况
print("\n" + "=" * 60)
print("3. 模拟理想情况 (pred>0.5)")
print("=" * 60)
pred_good = torch.ones_like(label) * 0.8  # 预测全是0.8
m_good = metrics(pred_good, label)
print("pred=0.8, label=1.0 -> metrics:", m_good.numpy())

print("\n修复完成！现在可以重新训练了。")
