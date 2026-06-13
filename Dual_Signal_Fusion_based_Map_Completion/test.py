import sys
sys.path.append('../')
sys.path.append('./')
from Dual_Signal_Fusion_based_Map_Completion.options.test_options import TestOptions
from Dual_Signal_Fusion_based_Map_Completion.models import create_model
from Dual_Signal_Fusion_based_Map_Completion.data_loader import get_data_loader_multistage
from Dual_Signal_Fusion_based_Map_Completion.utils.visualizer import save_images
from Dual_Signal_Fusion_based_Map_Completion.utils import html
import os
import numpy as np
from datetime import datetime
import torch


class Tester:
    def __init__(self, opt, model, test_dl):
        self.opt = opt
        self.model = model
        self.test_dl = test_dl

        # 1. 定义测试结果和日志保存的目录
        self.res_dir = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.epoch))
        os.makedirs(self.res_dir, exist_ok=True)
        self.log_file = os.path.join(self.res_dir, 'test_log.txt')

        # 2. 初始化日志文件
        with open(self.log_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"DelvMap Testing Log - Experiment: {opt.name} | Phase: {opt.phase} | Epoch: {opt.epoch}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

    def log(self, message, print_also=True):
        """辅助方法：将信息写入日志文件，并选择性地在终端打印"""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        if print_also:
            print(message)
    
    def pred(self):
        self.model.eval()
        # create a website
        web_dir = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.epoch))  # define the website directory
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        
        # 初始化所有的累加器
        tot_loss = 0.0
        tot_loss_traj = 0.0
        tot_loss_bldg = 0.0
        tot_loss_src = 0.0

        tot_metrics = np.zeros(4)   # 主路网 (Traj) 的指标
        tot_metrics1 = np.zeros(4)  # 建筑物 (Building) 的指标
        tot_metrics2 = np.zeros(4)  # 源图辅助路网 (Src) 的指标

        # 加入 no_grad 节省显存，加速推理
        with torch.no_grad():
            for i, data in enumerate(self.test_dl):
                self.model.set_input_test(data)
                
                # 4. 完整解包 7 个返回值
                (iter_loss, iter_metrics, iter_metrics1, iter_metrics2, 
                 loss_traj, loss_bldg, loss_src) = self.model.test()
                
                # 累加 Loss
                tot_loss += iter_loss.item()
                tot_loss_traj += loss_traj.item()
                tot_loss_bldg += loss_bldg.item()
                tot_loss_src += loss_src.item()
                
                # 累加 Metrics (转为 numpy)
                tot_metrics += iter_metrics.cpu().numpy()
                tot_metrics1 += iter_metrics1.cpu().numpy()
                tot_metrics2 += iter_metrics2.cpu().numpy()

                visuals = self.model.get_current_visuals()  # 获取当前生成的图像
                img_path = self.model.get_image_paths()     # 获取当前处理的图像路径
                
                # 将处理进度写入日志
                self.log(f'Processing: {img_path[0]}')
                
                # 保存图像 (这里的打印内容依然保留在终端，不会进我们自定义的 log 文件)
                save_images(webpage, visuals, img_path, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize)
        
        # 5. 计算整个测试集的平均值
        num_batches = len(self.test_dl)
        tot_loss /= num_batches
        tot_loss_traj /= num_batches
        tot_loss_bldg /= num_batches
        tot_loss_src /= num_batches
        
        tot_metrics /= num_batches
        tot_metrics1 /= num_batches
        tot_metrics2 /= num_batches

        # 6. 格式化最终的输出信息
        summary_msg = (
            f"\n{'='*70}\n"
            f"Test Summary (Total {num_batches} batches)\n"
            f"{'='*70}\n"
            f"Overall Loss: {tot_loss:.6f}\n"
            f"  -> Traj Loss:  {tot_loss_traj:.6f}\n"
            f"  -> Bldg Loss:  {tot_loss_bldg:.6f}\n"
            f"  -> Src Loss:   {tot_loss_src:.6f}\n\n"
            f"Metrics (Precision, Recall, F1, IOU):\n"
            f"  -> Traj (Main): P: {tot_metrics[0]:.6f} | R: {tot_metrics[1]:.6f} | F1: {tot_metrics[2]:.6f} | IOU: {tot_metrics[3]:.6f}\n"
            f"  -> Bldg (Mask): P: {tot_metrics1[0]:.6f} | R: {tot_metrics1[1]:.6f} | F1: {tot_metrics1[2]:.6f} | IOU: {tot_metrics1[3]:.6f}\n"
            f"  -> Src (Aux):   P: {tot_metrics2[0]:.6f} | R: {tot_metrics2[1]:.6f} | F1: {tot_metrics2[2]:.6f} | IOU: {tot_metrics2[3]:.6f}\n"
            f"{'='*70}\n"
        )
        
        # 输出 summary 并保存 HTML
        self.log(summary_msg)
        webpage.save()


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    test_dl = get_data_loader_multistage(opt.dataroot, 'test')
    tester = Tester(opt, model, test_dl)
    tester.pred()
