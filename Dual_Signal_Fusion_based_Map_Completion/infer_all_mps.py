"""DSFNet 全量推理 (Apple MPS 版)。

与 infer_all.py 功能一致，但把 net 搬到 MPS (Apple GPU)，比纯 CPU 快 ~60x。
MPS 不可用时自动回退 CPU。

输出：
    results/{name}/all_{epoch}/images_full/{idx}_pred_traj_img.png
    results/{name}/all_{epoch}/images_full/{idx}_pred_src_traj_img.png
    results/{name}/all_{epoch}/infer_all_log.txt

用法：
    python Dual_Signal_Fusion_based_Map_Completion/infer_all_mps.py \
        --name delvmap_exp2 --dataroot ./dataset/xian_2019_delvmap/ \
        --epoch 357 --model DSFNet --net_trans DSFNet --train_pattern DSFNet \
        --gpu_ids -1
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


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class FullInfererMPS:
    def __init__(self, opt, model, dl, device):
        self.opt = opt
        self.model = model
        self.dl = dl
        self.device = device

        self.res_dir = os.path.join(self.opt.results_dir, self.opt.name, 'all_%s' % self.opt.epoch)
        self.img_dir = os.path.join(self.res_dir, 'images_full')
        os.makedirs(self.img_dir, exist_ok=True)
        self.log_file = os.path.join(self.res_dir, 'infer_all_log.txt')

        with open(self.log_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"DelvMap Full Inference (MPS) - Exp: {opt.name} | Epoch: {opt.epoch}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Out: {self.img_dir}\n")
            f.write("=" * 70 + "\n\n")

    def log(self, msg, print_also=True):
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")
        if print_also:
            print(msg)

    @staticmethod
    def _to_gray_uint8(img_hwc3):
        if img_hwc3.ndim == 3:
            img = img_hwc3[:, :, 0]
        else:
            img = img_hwc3
        return img.astype(np.uint8)

    def run(self):
        # 把 net 搬到 MPS (绕过 BaseModel 的 cpu/cuda device 逻辑)
        net = self.model.netDSFNet
        net.to(self.device)
        net.eval()
        self.model.device = self.device

        n = len(self.dl)
        self.log(f"Total batches: {n}  device: {self.device}")

        t_start = datetime.now()
        with torch.no_grad():
            for i, data in enumerate(self.dl):
                img = data['traj_data'].to(self.device)
                src = data['src_data'].to(self.device)

                tr_out, sb_out, sr_out = net(img, src)

                # pred2im 等价：threshold 0.5 -> 0/1 -> *255，取单通道
                # 直接在 tensor 上做，省掉 model.pred_traj_img 的 numpy 往返
                def _bin255(t):
                    t0 = t[0, 0].detach().float().cpu().numpy()
                    return ((t0 > 0.5).astype(np.uint8) * 255)

                pred_traj = _bin255(tr_out)
                pred_src_traj = _bin255(sr_out)

                traj_path = data['traj_path'][0]
                idx = os.path.splitext(os.path.basename(traj_path))[0]

                cv2.imwrite(os.path.join(self.img_dir, f'{idx}_pred_traj_img.png'), pred_traj)
                cv2.imwrite(os.path.join(self.img_dir, f'{idx}_pred_src_traj_img.png'), pred_src_traj)

                if (i + 1) % 100 == 0 or i == n - 1:
                    elapsed = (datetime.now() - t_start).total_seconds()
                    rate = (i + 1) / elapsed
                    eta = (n - i - 1) / rate if rate > 0 else 0
                    self.log(f"  [{i+1}/{n}] saved idx={idx}  "
                             f"({rate:.1f} patch/s, ETA {eta/60:.1f} min)")

        elapsed = (datetime.now() - t_start).total_seconds()
        self.log(f"\nDone. {n} patches -> {self.img_dir}  ({elapsed/60:.1f} min, "
                 f"{n/elapsed:.1f} patch/s)")


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    dl = get_data_loader_multistage(opt.dataroot, 'all')
    device = pick_device()
    print(f"[device] using {device}")
    FullInfererMPS(opt, model, dl, device).run()
