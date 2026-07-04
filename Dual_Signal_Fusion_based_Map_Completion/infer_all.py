"""DSFNet 全量推理：把 train+val+test 所有 patch 都跑一遍，
   只保存预测掩膜，不计 metrics。

输出：
    results/{name}/all_{epoch}/images_full/{idx}_pred_traj_img.png
    results/{name}/all_{epoch}/images_full/{idx}_pred_src_traj_img.png
    results/{name}/all_{epoch}/infer_all_log.txt

用法（与 test.py 完全对称）：
    python Dual_Signal_Fusion_based_Map_Completion/infer_all.py \
        --name delvmap_exp2 --dataroot ./dataset/xian_2019_delvmap/ \
        --epoch 357 --model DSFNet --net_trans DSFNet --train_pattern DSFNet
"""
import sys
sys.path.append('../')
sys.path.append('./')
from Dual_Signal_Fusion_based_Map_Completion.options.test_options import TestOptions
from Dual_Signal_Fusion_based_Map_Completion.models import create_model
from Dual_Signal_Fusion_based_Map_Completion.data_loader import get_data_loader_multistage
import os
import cv2
import numpy as np
from datetime import datetime
import torch


class FullInferer:
    def __init__(self, opt, model, dl):
        self.opt = opt
        self.model = model
        self.dl = dl

        self.res_dir = os.path.join(self.opt.results_dir, self.opt.name, 'all_%s' % self.opt.epoch)
        self.img_dir = os.path.join(self.res_dir, 'images_full')
        os.makedirs(self.img_dir, exist_ok=True)
        self.log_file = os.path.join(self.res_dir, 'infer_all_log.txt')

        with open(self.log_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"DelvMap Full Inference Log - Experiment: {opt.name} | Epoch: {opt.epoch}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output dir: {self.img_dir}\n")
            f.write("=" * 70 + "\n\n")

    def log(self, message, print_also=True):
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        if print_also:
            print(message)

    @staticmethod
    def _to_gray_uint8(img_hwc3):
        """pred2im 输出 (H,W,3) 的 0/255 uint8（三通道复制），取一个通道就够。"""
        if img_hwc3.ndim == 3:
            img = img_hwc3[:, :, 0]
        else:
            img = img_hwc3
        return img.astype(np.uint8)

    def run(self):
        self.model.eval()
        n = len(self.dl)
        self.log(f"Total batches to infer: {n}")
        with torch.no_grad():
            for i, data in enumerate(self.dl):
                self.model.set_input_test(data)
                # forward 即可，不需要算 loss/metrics
                self.model.forward()

                # 用 traj_path 的 basename 当 patch 索引（如 '123.npy' -> '123'）
                traj_path = data['traj_path'][0]
                idx = os.path.splitext(os.path.basename(traj_path))[0]

                pred_traj = self._to_gray_uint8(self.model.pred_traj_img)
                pred_src_traj = self._to_gray_uint8(self.model.pred_src_traj_img)

                cv2.imwrite(os.path.join(self.img_dir, f'{idx}_pred_traj_img.png'), pred_traj)
                cv2.imwrite(os.path.join(self.img_dir, f'{idx}_pred_src_traj_img.png'), pred_src_traj)

                if (i + 1) % 50 == 0 or i == n - 1:
                    self.log(f"  [{i+1}/{n}] saved idx={idx}")

        self.log(f"\nFull inference done. {n} patches → {self.img_dir}")


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    dl = get_data_loader_multistage(opt.dataroot, 'all')
    FullInferer(opt, model, dl).run()
