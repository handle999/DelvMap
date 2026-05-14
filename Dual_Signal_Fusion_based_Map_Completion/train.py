import sys
import numpy as np
sys.path.append('../')
sys.path.append('./')

from Dual_Signal_Fusion_based_Map_Completion.options.train_options import TrainOptions
from Dual_Signal_Fusion_based_Map_Completion.utils.visualizer import Visualizer
from Dual_Signal_Fusion_based_Map_Completion.models import create_model
from Dual_Signal_Fusion_based_Map_Completion.data_loader import get_data_loader_multistage
from datetime import datetime
from tqdm import tqdm
import os
import torch
import time


class MultiTrainer:
    def __init__(self, opt, model, train_dl, val_dl, visualizer):
        self.opt = opt
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.visualizer = visualizer

        # 设置日志文件
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_file = os.path.join(self.log_dir, 'training.log')

        # 初始化日志文件
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("DelvMap DSFNet Training Log\n")
            f.write("Start Time: %s\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            f.write("=" * 70 + "\n\n")

        # FIFO checkpoint队列，保留最近的N个最佳模型
        self.max_keep_checkpoints = opt.max_keep_checkpoints
        self.saved_checkpoints = []  # 存储格式: [(epoch, f1_score), ...]

    def log(self, message, print_also=True):
        """写入日志文件并在终端打印"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
        if print_also:
            print(message)

    def cleanup_old_checkpoints(self, keep_latest=True):
        """FIFO清理：保留最近的N个checkpoint，删除更早的"""
        import glob as glob_module

        checkpoint_dir = self.log_dir
        pattern = os.path.join(checkpoint_dir, "*_net_DSFNet.pth")
        all_checkpoints = glob_module.glob(pattern)

        if len(all_checkpoints) <= self.max_keep_checkpoints:
            self.log(f"[FIFO Cleanup] max_keep={self.max_keep_checkpoints}, current={len(all_checkpoints)}, no cleanup needed")
            return  # 不需要清理

        # 获取所有带epoch编号的checkpoint文件
        epoch_checkpoints = []
        for ckpt_path in all_checkpoints:
            filename = os.path.basename(ckpt_path)
            if filename == "latest_net_DSFNet.pth":
                continue  # 跳过latest
            # 提取epoch编号
            try:
                epoch_num = int(filename.split('_')[0])
                epoch_checkpoints.append((epoch_num, ckpt_path))
            except (ValueError, IndexError):
                continue

        # 按epoch排序
        epoch_checkpoints.sort(key=lambda x: x[0])

        # 需要删除的：保留最新的N个，其余删除
        to_keep = epoch_checkpoints[-self.max_keep_checkpoints:]
        to_delete = epoch_checkpoints[:-self.max_keep_checkpoints]

        if to_delete:
            self.log(f"\n[FIFO Cleanup] Keeping {self.max_keep_checkpoints} latest checkpoints, deleting {len(to_delete)} old ones...")
            for epoch_num, ckpt_path in to_delete:
                try:
                    os.remove(ckpt_path)
                    self.log(f"  Deleted: {os.path.basename(ckpt_path)}")
                except OSError as e:
                    self.log(f"  Failed to delete {ckpt_path}: {e}")

    def fit(self):
        best_f1_score = 0.0
        early_stopping = EarlyStopper(patience=30, min_delta=0)

        for epoch in range(1, self.opt.n_epochs + 1):
            epoch_start = datetime.now()

            self.log(f"\n{'='*60}")
            self.log(f"Epoch {epoch}/{self.opt.n_epochs}")
            self.log(f"{'='*60}")

            # ========== 训练阶段 (带tqdm) ==========
            self.model.train()
            train_tot_loss = 0
            train_tot_loss_traj = 0
            train_tot_loss_building = 0
            train_tot_loss_src_traj = 0
            train_tot_metrics = np.zeros(4)  # [precision, recall, f1, iou]
            train_tot_metrics1 = np.zeros(4)
            train_tot_metrics2 = np.zeros(4)

            pbar_train = tqdm(self.train_dl, desc=f"Train E{epoch}",
                            leave=False, ncols=100)

            for i, data in enumerate(pbar_train):
                self.model.set_input(data)
                iter_loss, iter_metrics, iter_metrics1, iter_metrics2, loss_building, loss_traj, loss_src_traj = self.model.optimize_parameters()
                # 先转到CPU再转numpy，避免GPU tensor的问题
                iter_metrics = iter_metrics.cpu().numpy()
                iter_metrics1 = iter_metrics1.cpu().numpy()
                iter_metrics2 = iter_metrics2.cpu().numpy()

                train_tot_loss += iter_loss.item()
                train_tot_loss_traj += loss_traj.item()
                train_tot_loss_building += loss_building.item()
                train_tot_loss_src_traj += loss_src_traj.item()
                train_tot_metrics += iter_metrics
                train_tot_metrics1 += iter_metrics1
                train_tot_metrics2 += iter_metrics2

                # 更新tqdm显示
                pbar_train.set_postfix({
                    'Loss': f"{iter_loss.item():.6f}",
                    'Traj Loss': f"{loss_traj.item():.6f}",
                    'Bldg Loss': f"{loss_building.item():.6f}",
                    'Src Loss': f"{loss_src_traj.item():.6f}",
                    'P': f"{iter_metrics[0]:.6f}",
                    'R': f"{iter_metrics[1]:.6f}",
                    'F1': f"{iter_metrics[2]:.6f}",
                    'IOU': f"{iter_metrics[3]:.6f}"
                })

            pbar_train.close()

            # 计算训练集平均metrics
            n_train_batches = len(self.train_dl)
            train_avg_loss = train_tot_loss / n_train_batches
            train_avg_loss_traj = train_tot_loss_traj / n_train_batches
            train_avg_loss_building = train_tot_loss_building / n_train_batches
            train_avg_loss_src_traj = train_tot_loss_src_traj / n_train_batches
            train_avg_metrics = train_tot_metrics / n_train_batches
            train_avg_metrics1 = train_tot_metrics1 / n_train_batches
            train_avg_metrics2 = train_tot_metrics2 / n_train_batches

            train_msg = (
                f"\n[Train Summary] Loss: {train_avg_loss:.6f} | "
                f"Traj Loss: {train_avg_loss_traj:.6f} | "
                f"Bldg Loss: {train_avg_loss_building:.6f} | "
                f"Src Loss: {train_avg_loss_src_traj:.6f} | "
                f"Traj: P={train_avg_metrics[0]:.6f} R={train_avg_metrics[1]:.6f} F1={train_avg_metrics[2]:.6f} IOU={train_avg_metrics[3]:.6f} | "
                f"Bldg: P={train_avg_metrics1[0]:.6f} R={train_avg_metrics1[1]:.6f} F1={train_avg_metrics1[2]:.6f} IOU={train_avg_metrics1[3]:.6f} | "
                f"Src: P={train_avg_metrics2[0]:.6f} R={train_avg_metrics2[1]:.6f} F1={train_avg_metrics2[2]:.6f} IOU={train_avg_metrics2[3]:.6f}"
            )
            self.log(train_msg)

            # ========== 验证阶段 (带tqdm) ==========
            self.log(f"\n[Validation] Running on {len(self.val_dl)} samples...")
            self.model.eval()
            val_tot_loss = 0
            val_tot_loss_traj = 0
            val_tot_loss_building = 0
            val_tot_loss_src_traj = 0
            val_tot_metrics = np.zeros(4)
            val_tot_metrics1 = np.zeros(4)
            val_tot_metrics2 = np.zeros(4)

            pbar_val = tqdm(self.val_dl, desc=f"Val   E{epoch}",
                          leave=False, ncols=100)

            with torch.no_grad():
                for i, data in enumerate(pbar_val):
                    self.model.set_input(data)
                    iter_loss, iter_metrics, iter_metrics1, iter_metrics2, loss_building, loss_traj, loss_src_traj = self.model.test()

                    val_tot_loss += iter_loss.item()
                    val_tot_loss_traj += loss_traj.item()
                    val_tot_loss_building += loss_building.item()
                    val_tot_loss_src_traj += loss_src_traj.item()
                    # 先转到CPU再转numpy，避免GPU tensor直接转numpy的问题
                    val_tot_metrics += iter_metrics.cpu().numpy()
                    val_tot_metrics1 += iter_metrics1.cpu().numpy()
                    val_tot_metrics2 += iter_metrics2.cpu().numpy()

                    # 更新tqdm显示
                    pbar_val.set_postfix({
                        'Loss': f"{iter_loss.item():.6f}",
                        'Traj Loss': f"{loss_traj.item():.6f}",
                        'Bldg Loss': f"{loss_building.item():.6f}",
                        'Src Loss': f"{loss_src_traj.item():.6f}",
                        'P': f"{iter_metrics.numpy()[0]:.6f}",
                        'R': f"{iter_metrics.numpy()[1]:.6f}",
                        'F1': f"{iter_metrics.numpy()[2]:.6f}",
                        'IOU': f"{iter_metrics.numpy()[3]:.6f}"
                    })

            pbar_val.close()

            # 计算验证集平均metrics
            n_val_batches = len(self.val_dl)
            val_avg_loss = val_tot_loss / n_val_batches
            val_avg_loss_traj = val_tot_loss_traj / n_val_batches
            val_avg_loss_building = val_tot_loss_building / n_val_batches
            val_avg_loss_src_traj = val_tot_loss_src_traj / n_val_batches
            val_avg_metrics = val_tot_metrics / n_val_batches
            val_avg_metrics1 = val_tot_metrics1 / n_val_batches
            val_avg_metrics2 = val_tot_metrics2 / n_val_batches

            val_msg = (
                f"\n[Val Summary] Loss: {val_avg_loss:.6f} | "
                f"Traj Loss: {val_avg_loss_traj:.6f} | "
                f"Bldg Loss: {val_avg_loss_building:.6f} | "
                f"Src Loss: {val_avg_loss_src_traj:.6f} | "
                f"Traj: P={val_avg_metrics[0]:.6f} R={val_avg_metrics[1]:.6f} F1={val_avg_metrics[2]:.6f} IOU={val_avg_metrics[3]:.6f} | "
                f"Bldg: P={val_avg_metrics1[0]:.6f} R={val_avg_metrics1[1]:.6f} F1={val_avg_metrics1[2]:.6f} IOU={val_avg_metrics1[3]:.6f} | "
                f"Src: P={val_avg_metrics2[0]:.6f} R={val_avg_metrics2[1]:.6f} F1={val_avg_metrics2[2]:.6f} IOU={val_avg_metrics2[3]:.6f}"
            )
            self.log(val_msg)

            # ========== 保存模型 (根据Src的F1)，保留最近5个最佳) ==========
            current_f1 = val_avg_metrics2[2]  # Src的F1分数
            save_flag = False
            # 只有当F1>0时才保存（避免初始F1=0时一直保存）
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                save_flag = True
                self.log(f'\n>>> Best F1 Score: {best_f1_score:.6f} (epoch {epoch}) - Saving model...')
                self.model.save_networks('latest')
                self.model.save_networks(epoch)

                # FIFO清理旧checkpoint
                self.cleanup_old_checkpoints()
            elif current_f1 > 0:
                # F1>0但不是最佳，也更新best（用于记录）
                pass

            # ========== 保存结果到文件 ==========
            results_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'results.txt')
            with open(results_path, 'a') as f:
                f.write(f"===== Epoch {epoch} =====\n")
                f.write(f"Train Loss: {train_avg_loss:.6f}\n")
                f.write(f"Train Traj Loss: {train_avg_loss_traj:.6f}\n")
                f.write(f"Train Bldg Loss: {train_avg_loss_building:.6f}\n")
                f.write(f"Train Src Loss: {train_avg_loss_src_traj:.6f}\n")
                f.write(f"Traj - P:{train_avg_metrics[0]:.6f} R:{train_avg_metrics[1]:.6f} F1:{train_avg_metrics[2]:.6f} IOU:{train_avg_metrics[3]:.6f}\n")
                f.write(f"Bldg - P:{train_avg_metrics1[0]:.6f} R:{train_avg_metrics1[1]:.6f} F1:{train_avg_metrics1[2]:.6f} IOU:{train_avg_metrics1[3]:.6f}\n")
                f.write(f"Src  - P:{train_avg_metrics2[0]:.6f} R:{train_avg_metrics2[1]:.6f} F1:{train_avg_metrics2[2]:.6f} IOU:{train_avg_metrics2[3]:.6f}\n")
                f.write(f"Val Loss: {val_avg_loss:.6f}\n")
                f.write(f"Val Traj Loss: {val_avg_loss_traj:.6f}\n")
                f.write(f"Val Bldg Loss: {val_avg_loss_building:.6f}\n")
                f.write(f"Val Src Loss: {val_avg_loss_src_traj:.6f}\n")
                f.write(f"Traj - P:{val_avg_metrics[0]:.6f} R:{val_avg_metrics[1]:.6f} F1:{val_avg_metrics[2]:.6f} IOU:{val_avg_metrics[3]:.6f}\n")
                f.write(f"Bldg - P:{val_avg_metrics1[0]:.6f} R:{val_avg_metrics1[1]:.6f} F1:{val_avg_metrics1[2]:.6f} IOU:{val_avg_metrics1[3]:.6f}\n")
                f.write(f"Src  - P:{val_avg_metrics2[0]:.6f} R:{val_avg_metrics2[1]:.6f} F1:{val_avg_metrics2[2]:.6f} IOU:{val_avg_metrics2[3]:.6f}\n")
                if save_flag:
                    f.write(f"*** BEST MODEL SAVED ***\n")
                f.write("\n")

            # ========== 可视化 ==========
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, True)

            # ========== 早停检查 ==========
            if early_stopping.early_stop(current_f1):
                self.log(f'\n>>> Early stopping triggered at epoch {epoch}')
                break

            # ========== 学习率更新 ==========
            self.model.update_learning_rate()

            epoch_time = datetime.now() - epoch_start
            self.log(f'\nTime cost: {epoch_time}')
            self.log(f"Learning Rate: {self.model.optimizers[0].param_groups[0]['lr']:.6f}")

        self.log(f"\n{'='*60}")
        self.log(f"Training Complete! Best F1: {best_f1_score:.6f}")
        self.log(f"{'='*60}")


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_f1 = -np.inf

    def early_stop(self, validation_f1):
        if validation_f1 > self.max_validation_f1:
            self.max_validation_f1 = validation_f1
            self.counter = 0
        elif validation_f1 < (self.max_validation_f1 + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__ == '__main__':
    opt = TrainOptions().parse()
    print(opt)
    model = create_model(opt)
    model.setup(opt)

    if opt.train_pattern == 'DSFNet':
        train_dl = get_data_loader_multistage(opt.dataroot, 'train', opt.batch_size)
        val_dl = get_data_loader_multistage(opt.dataroot, 'val', opt.batch_size)
        print(f"\nLoaded: {len(train_dl)} train batches, {len(val_dl)} val batches")
        visualizer = Visualizer(opt)
        trainer = MultiTrainer(opt, model, train_dl, val_dl, visualizer)
        trainer.fit()
