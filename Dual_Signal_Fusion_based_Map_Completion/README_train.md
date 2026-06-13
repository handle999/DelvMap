# DSFNet 训练详细指南

## 1. 快速开始

### 训练命令
```bash
python Dual_Signal_Fusion_based_Map_Completion/train.py \
    --name Your_Task_Name \
    --dataroot /path/to/your/dataset \
    --lam 0.2 \
    --batch_size 8 \
    --train_pattern DSFNet \
    --net_trans DSFNet \
    --model DSFNet
```

### 推理命令
```bash
python Dual_Signal_Fusion_based_Map_Completion/test.py \
    --name Your_Task_Name \
    --dataroot /path/to/your/dataset \
    --lam 0.2 \
    --train_pattern DSFNet \
    --net_trans DSFNet \
    --model DSFNet \
    --epoch 100
```

---

## 2. 训练参数详解

### 2.1 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--name` | 实验名称，用于保存checkpoint和日志 | `experiment_name` |
| `--dataroot` | 数据集根目录（包含train/val或split_indices.json） | **必填** |
| `--gpu_ids` | GPU设备ID，多GPU用逗号分隔 | `0` |
| `--checkpoints_dir` | 保存模型的目录 | `./checkpoints` |
| `--batch_size` | 批大小 | `8` |
| `--num_threads` | 数据加载线程数 | `4` |

### 2.2 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型 | `Unet` |
| `--net_trans` | 网络架构 | `Unet` |
| `--input_nc` | 输入通道数 | 自动（DSFNet为5） |
| `--output_nc` | 输出通道数 | `1` |
| `--norm` | 归一化方式 | `batch` |
| `--init_type` | 权重初始化方式 | `kaiming` |
| `--init_gain` | 初始化增益 | `0.02` |
| `--no_dropout` | 是否禁用dropout | False |

**训练 DSFNet 模型时，需设置：**
```bash
--model DSFNet --net_trans DSFNet --train_pattern DSFNet
```

### 2.3 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--n_epochs` | 训练轮数 | `200` |
| `--lr` | 初始学习率 | `0.0005` |
| `--beta1` | Adam动量参数 | `0.99` |
| `--lr_policy` | 学习率衰减策略 | `step` |
| `--lr_decay_iters` | 学习率衰减间隔（epoch） | `50` |
| `--phase` | 训练/测试阶段 | `train` |

### 2.4 验证与早停参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample_interval` | 验证间隔（iterations） | `50` |
| `--lam` | 损失函数权重参数 | `0.2` |

**早停机制：**
- 在 `train.py` 中实现，patience=30
- 当验证F1分数连续30次不上升时停止训练

### 2.5 可视化参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--display_id` | visdom显示窗口ID，设为0可禁用 | `1` |
| `--display_freq` | 屏幕显示训练结果频率（iterations） | `50` |
| `--display_ncols` | visdom单行显示的图片数量 | `4` |
| `--display_server` | visdom服务器地址 | `http://localhost` |
| `--display_env` | visdom环境名称 | `main` |
| `--display_port` | visdom服务器端口 | `8097` |
| `--update_html_freq` | 保存HTML可视化频率 | `50` |
| `--print_freq` | 控制台打印频率 | `100` |
| `--display_winsize` | 显示窗口大小 | `256` |
| `--no_html` | 禁用HTML保存 | False |

### 2.6 模型保存参数

| 参数 | 说明 | 默认值 |
|--------------|------|--------|
| `--save_latest_freq` | 保存最新模型频率（iterations） | `50` |
| `--save_epoch_freq` | 保存epoch checkpoint频率 | `5` |
| `--save_by_iter` | 是否按iteration保存 | False |
| `--continue_train` | 是否继续训练（断点续训） | False |
| `--epoch` | 加载模型的epoch编号 | `latest` |
| `--load_iter` | 加载模型的iteration编号 | `0` |
| `--max_keep_checkpoints` | FIFO保留的checkpoint数量 | `5` |

---

## 3. 高级训练设置

### 3.1 断点续训

当训练中断后，需要继续训练时使用：

```bash
python Dual_Signal_Fusion_based_Map_Completion/train.py \
    --name delvmap_exp1 \
    --dataroot ./dataset/xian_2019_delvmap \
    --continue_train \
    --epoch latest \
    --gpu_ids 5 \
    --model DSFNet --net_trans DSFNet --train_pattern DSFNet
```

**说明：**
- `--continue_train`：启用续训模式
- `--epoch latest`：从最新保存的模型继续，也可指定具体epoch如`--epoch 100`
- 其他参数（batch_size等）应与原训练保持一致
- **学习率会自动从checkpoint加载**，无需手动指定
- 训练结果会**追加**到`results.txt`，不会覆盖

### 3.2 Checkpoint文件管理

模型文件命名规则：
```
{epoch}_net_DSFNet.pth    # 第XX个epoch的模型
latest_net_DSFNet.pth     # 最新模型（best F1对应的模型）
```

**关键特性：**
- 模型保存基于**Val Src F1**分数（仅使用卫星影像推理时使用）
- 每个epoch只保存一次best模型（当F1超过历史最佳时）
- 模型权重保存不受可视化失败影响
- **FIFO自动清理**：只保留最近N个最佳checkpoint（默认5个，可通过`--max_keep_checkpoints`配置），超出的会自动删除

**FIFO清理机制：**
- 当保存新模型后，若checkpoint数量超过`max_keep_checkpoints`，自动删除最早的
- `latest_net_DSFNet.pth` 始终保留（指向最佳模型）
- 日志中会显示删除的文件列表

**自定义保留数量：**
```bash
--max_keep_checkpoints 10   # 保留最近10个checkpoint
```

**示例 - 查看已保存的checkpoint：**
```bash
ls -lh checkpoints/delvm_exp1/*.pth
```

### 3.3 训练状态查看

**查看当前训练进度：**
```bash
# 查看最新训练结果
tail -20 checkpoints/delvm_exp1/results.txt

# 查看学习率变化
grep "Learning Rate:" checkpoints/delvm_exp1/training.log | tail -10

# 查看完整日志
cat checkpoints/delvm_exp1/training.log
```

**典型训练状态示例（Epoch 173）：**
```
Val Loss: 0.648231
Val Src - P:0.893389 R:0.857994 F1:0.875156 IOU:0.781696
*** BEST MODEL SAVED ***
Learning Rate: 0.000063
```

### 3.4 常见中断与恢复

| 中断原因 | 恢复方式 |
|----------|----------|
| 磁盘空间不足(可视化保存失败) | 模型权重已保存，使用`--continue_train --epoch latest`恢复 |
| GPU内存不足 | 减小`--batch_size` |
| SSH连接断开 | 使用`--continue_train`从latest恢复 |
| 手动停止 | 同上，从latest或指定epoch恢复 |

### 3.2 学习率调度

代码支持以下学习率策略：

```bash
--lr_policy step        # 默认，每50个epoch衰减一半
--lr_policy linear      # 线性衰减
--lr_policy plateau    # 根据验证损失调整
--lr_policy cosine      # 余弦退火
```

### 3.3 自定义训练参数示例

```bash
python Dual_Signal_Fusion_based_Map_Completion/train.py \
    --name my_experiment \
    --dataroot /data/delvmap_dataset \
    --batch_size 16 \
    --n_epochs 300 \
    --lr 0.001 \
    --lr_policy cosine \
    --sample_interval 100 \
    --display_id 0 \
    --no_html \
    --model DSFNet --net_trans DSFNet --train_pattern DSFNet
```

---

## 4. 输出文件结构

训练完成后，在 `--checkpoints_dir` 目录下会生成以下文件：

```
checkpoints/
└── Your_Task_Name/
    ├── DSFNet_net_DSFNet_XX.pth     # 第XX个epoch的模型
    ├── DSFNet_net_DSFNet_latest.pth # 最新模型
    ├── web/                         # HTML可视化
    │   ├── index.html               # 可视化网页
    │   └── images/                  # 中间结果图片
    ├── loss_log.txt                 # 训练 loss 日志
    ├── results.txt                  # 验证结果日志
    └── train_opt.txt                # 训练配置
```

### 4.1 日志文件说明

**loss_log.txt** - 训练过程中的loss记录：
```
(epoch: 1, iters: 0, time: 0.123, data: 0.045) all: 0.856
(epoch: 1, iters: 1, time: 0.098, data: 0.032) all: 0.742
...
```

**results.txt** - 验证集的指标记录：
```
epoch	1	iter	50	loss_Dice	0.123456
epoch	1	iter	50	loss_Dice	0.123456	precision	0.8500	recall	0.8200	f1	0.8350	cl_iou	0.7200
...
```

**指标含义：**
- `precision`: 精确率
- `recall`: 召回率
- `f1`: F1分数
- `cl_iou`: 类别IOU（道路）

### 4.2 HTML可视化

**单个epoch查看**：打开 `web/index.html` 可以看到结果。
- 输入轨迹图 (traj)
- 卫星影像 (src)
- 真实道路标签 (label)
- 预测道路 (pred_traj_img)
- 预测建筑 (pred_building_img)
- 仅用卫星影像预测的道路 (pred_src_traj_img)

**所有epoch汇总**：运行以下命令生成汇总页面：
```bash
python Dual_Signal_Fusion_based_Map_Completion/utils/generate_all_epochs_html.py
```
然后打开 `web/all_epochs.html`，包含：
- 所有172个epoch的完整展示
- 每行8列：Traj Input, Src Input, GT Road, Pred Road(Traj+Src), Satellite, GT Building, Pred Building, Pred Road(Src Only)
- 右侧固定导航：支持输入epoch编号快速跳转
- 链接返回最新epoch页面

**图片文件说明**：
- `epochXXX_img_1.png`: 输入轨迹图 (traj)
- `epochXXX_img_2.png`: 输入卫星图 (src)
- `epochXXX_label.png`: 真实道路标签 (label)
- `epochXXX_pred_traj_img.png`: 预测道路 (traj+src融合)
- `epochXXX_src.png`: 卫星影像 (src)
- `epochXXX_building_label.png`: 建筑标签
- `epochXXX_pred_building_img.png`: 预测建筑
- `epochXXX_pred_src_traj_img.png`: 仅用卫星影像预测的道路

---

## 5. 数据集准备

### 5.1 数据集目录结构

```
dataset_root/
├── split_indices.json        # 数据集划分索引
├── traj_and_point_split/     # 轨迹+点数据 (.npy)
├── label/                    # 道路标签 (.png)
├── src_split/                # 卫星影像 (.png)
└── building_label/           # 建筑标签 (.png)
```

### 5.2 split_indices.json 格式

```json
{
    "train": [0, 1, 2, ..., 799],
    "val": [800, 801, ..., 899],
    "test": [900, 901, ..., 999]
}
```

默认划分比例：80%训练 / 10%验证 / 10%测试

### 5.3 数据格式

- **traj_and_point_split/*.npy**: 2通道轨迹数据 (shape: HxWx2)
- **src_split/*.png**: 3通道RGB卫星影像
- **label/*.png**: 1通道道路标签 (二值化)
- **building_label/*.png**: 1通道建筑标签 (二值化)

---

## 6. 模型架构

DSFNet (Dual Signal Fusion Network) 包含三个输出头：

1. **道路预测 (traj)**: 融合轨迹和卫星影像预测道路
2. **建筑预测 (building)**: 从卫星影像预测建筑
3. **仅影像道路预测 (src_pred_traj)**: 仅用卫星影像预测道路

### 损失函数

- 道路预测: `BCEDiceLoss` (BCE + Dice Loss)
- 建筑预测: `BCEWithLogitsLoss`
- 仅影像道路: `BCEDiceLoss`

总损失 = `loss_Traj + loss_Building + loss_src_Traj`

---

## 7. 常见问题

### Q1: 显存不足怎么办？
```bash
# 减小batch_size
--batch_size 4
# 或使用CPU（不推荐）
--gpu_ids -1
```

### Q2: 如何使用多GPU训练？
```bash
--gpu_ids 0,1,2,3
```

### Q3: 如何查看训练日志？
```bash
# 查看loss日志
cat checkpoints/Your_Task_Name/loss_log.txt

# 查看验证结果
cat checkpoints/Your_Task_Name/results.txt
```

### Q4: 如何从指定epoch继续？
```bash
--continue_train --epoch 50
```

### Q5: 训练速度太慢？
```bash
# 减少验证频率
--sample_interval 200
# 禁用可视化
--display_id 0 --no_html
```

---

## 8. 完整示例

### 示例1: 从头训练
```bash
python Dual_Signal_Fusion_based_Map_Completion/train.py \
    --name delvmap_exp2 \
    --dataroot ./dataset/xian_2019_delvmap/ \
    --gpu_ids 5 \
    --max_keep_checkpoints 10 \
    --batch_size 16 \
    --n_epochs 500 \
    --lr 0.0005 \
    --model DSFNet --net_trans DSFNet --train_pattern DSFNet
```

### 示例2: 断点续训
```bash
python Dual_Signal_Fusion_based_Map_Completion/train.py \
    --name delvmap_exp1 \
    --dataroot /data/delvmap_dataset \
    --continue_train \
    --epoch latest \
    --model DSFNet --net_trans DSFNet --train_pattern DSFNet
```

### 示例3: 推理预测
```bash
python Dual_Signal_Fusion_based_Map_Completion/test.py \
    --name delvmap_exp1 \
    --dataroot /data/delvmap_dataset \
    --epoch 100 \
    --model DSFNet --net_trans DSFNet --train_pattern DSFNet
```

---

## 9. 依赖环境

主要依赖包：
- torch >= 2.0
- torchvision
- numpy
- opencv-python
- tqdm
- Pillow
- scipy
- scikit-learn

建议使用 conda 环境安装：
```bash
conda create -n dsfnet python=3.10
conda activate dsfnet
pip install torch torchvision numpy opencv-python tqdm Pillow scipy scikit-learn
```

---

## 10. 训练日志与可视化

### 10.1 日志文件

训练过程中会在 `checkpoints/Your_Task_Name/` 目录下生成以下日志文件：

| 文件 | 说明 |
|------|------|
| `training.log` | 完整训练日志（推荐查看），包含时间戳、Loss、P、R、F1、IOU |
| `results.txt` | 每个epoch的验证指标记录 |
| `web/index.html` | HTML可视化网页 |

### 10.2 tqdm 进度条

训练和验证阶段都会显示 tqdm 进度条：

```
Train E1: 100%|██████████| 100/100 [00:45<00:00, Loss:0.8532, P:0.123, R:0.089, F1:0.102, IOU:0.054]

[Train Summary] Loss: 0.8532 | Traj: P=0.1234 R=0.0891 F1=0.1023 IOU=0.0542 | Bldg: P=0.2341 R=0.1823 F1=0.2054 IOU=0.1132 | Src: P=0.1567 R=0.1234 F1=0.1389 IOU=0.0756

[Validation] Running on 229 samples...
Val   E1: 100%|██████████| 229/229 [00:12<0:00, Loss:0.6521, P:0.234, R:0.189, F1:0.208, IOU=0.116]

[Val Summary] Loss: 0.6521 | Traj: P=0.2345 R=0.1892 F1=0.2089 IOU=0.1165 | Bldg: P=0.3456 R=0.2876 F1=0.3142 IOU=0.1876 | Src: P=0.2789 R=0.2345 F1=0.2551 IOU=0.1456
```

- **Train/Val进度条**: 显示当前epoch的进度
- **postfix**: 实时显示当前batch的Loss、P、R、F1、IOU
- **Summary**: 每个epoch结束后显示该epoch的平均指标

### 10.3 三个输出头的指标

| 输出头 | 说明 | 标签 |
|--------|------|------|
| **Traj** | 轨迹+卫星影像融合预测道路 | label (道路真值) |
| **Bldg** | 建筑预测 | building_label (建筑真值) |
| **Src** | 仅用卫星影像预测道路 | label (道路真值) |

模型保存逻辑：使用 **Src** (仅卫星影像) 的 **F1分数** 作为保存标准

---

## 11. 常见问题

### Q1: 训练时F1/IOU显示为0

**原因**: 数据预处理错误，label值不在[0,1]范围

**解决**: 确保 `data_loader.py` 中 label 处理正确：
```python
label = img_transform(label).float()  # 不要 /255.0
```

### Q2: 如何查看训练日志

```bash
# 查看完整日志
cat checkpoints/Your_Task_Name/training.log

# 实时查看日志
tail -f checkpoints/Your_Task_Name/training.log
```

### Q3: 训练时RuntimeError: tensor size mismatch

**错误信息**:
```
RuntimeError: The size of tensor a (64) must match the size of tensor b (2) at non-singleton dimension 1
```

**原因**: [translator.py:329](Dual_Signal_Fusion_based_Map_Completion/models/translator.py#L329) 中 `temp` 张量维度与 `max_channel` 不匹配

**解决**: 修复 `translator.py` 第329行：
```python
# 原代码 (错误)
temp = torch.ones_like(sb_out1[:, 0, :, :])  # [B, H, W]

# 修复后
temp = torch.ones_like(sb_out1[:, :1, :, :])  # [B, 1, H, W]
```

### Q4: 训练初期F1一直为0，正常吗？

**现象**: 训练前几个epoch，F1/P/R/IOU都很低，甚至为0

**分析**: 这是正常现象，原因如下：
1. 模型权重随机初始化，预测结果接近随机
2. 训练初期模型正在学习特征表达
3. 使用CPU小样本测试验证，20步后F1可从0.088提升到0.350

**判断标准**:
- 如果Loss在下降，说明训练正常进行
- 如果F1从低到高逐步提升，说明模型在学习
- 预期：10-20个epoch后F1应有明显提升

### Q5: CPU测试验证方法

如需在CPU上验证训练是否正常工作：

```bash
# 运行详细调试脚本
python debug_train_cpu.py      # 单样本详细分析
python debug_train_epoch.py    # 多epoch训练测试
python debug_predistribution.py # 预测分布分析
```

这些脚本会打印：
- 原始数据值范围
- DataLoader处理后的值
- 模型输出的分布
- 训练过程中的Loss和F1变化

### Q6: 如何排查训练问题

1. **先运行CPU测试验证**:
   ```bash
   python debug_train_cpu.py
   ```
   确认数据加载和模型前向正常

2. **检查日志文件**:
   ```bash
   cat checkpoints/Your_Task_Name/training.log
   ```

3. **检查Loss是否下降**: 如果Loss不下降，可能是模型结构或数据问题

4. **检查F1变化趋势**: 初期F1低是正常的，关键是看是否逐步提升
