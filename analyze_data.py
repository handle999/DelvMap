import sys
sys.path.append('./')
import numpy as np
import cv2
import torchvision.transforms as transforms
import os

sample_idx = "0"
data_root = "dataset/xian_2019_delvmap/"

print("=" * 70)
print("RAW FILE CHECK")
print("=" * 70)

# 1. traj npy file
traj_path = os.path.join(data_root, "traj_and_point_split", sample_idx + ".npy")
traj_raw = np.load(traj_path)
print("\n[1] traj_and_point_split/*.npy")
print("    shape:", traj_raw.shape)
print("    dtype:", traj_raw.dtype)
print("    min:", traj_raw.min())
print("    max:", traj_raw.max())
print("    unique:", np.unique(traj_raw)[:5])

# 2. src png file
src_path = os.path.join(data_root, "src_split", sample_idx + ".png")
src_raw = cv2.imread(src_path)
print("\n[2] src_split/*.png")
print("    shape:", src_raw.shape)
print("    dtype:", src_raw.dtype)
print("    min:", src_raw.min())
print("    max:", src_raw.max())

# 3. label png file
label_path = os.path.join(data_root, "label", sample_idx + ".png")
label_raw = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
print("\n[3] label/*.png")
print("    shape:", label_raw.shape)
print("    dtype:", label_raw.dtype)
print("    min:", label_raw.min())
print("    max:", label_raw.max())
print("    unique:", np.unique(label_raw))

# 4. building png file
building_path = os.path.join(data_root, "building_label", sample_idx + ".png")
building_raw = cv2.imread(building_path, cv2.IMREAD_GRAYSCALE)
print("\n[4] building_label/*.png")
print("    shape:", building_raw.shape)
print("    dtype:", building_raw.dtype)
print("    min:", building_raw.min())
print("    max:", building_raw.max())
print("    unique:", np.unique(building_raw))

print("\n" + "=" * 70)
print("DATALOADER PROCESSING")
print("=" * 70)

transform = transforms.ToTensor()

# traj processing
print("\n--- TRAJ ---")
traj_s1 = np.asarray(np.load(traj_path))
traj_s1 = np.array(traj_s1, dtype="float")
print("Step1: np.asarray + dtype=float")
print("  min: %.2f, max: %.2f" % (traj_s1.min(), traj_s1.max()))

traj_s2 = transform(np.array(traj_s1, dtype="uint8")).float()
print("Step2: ToTensor() + float")
print("  shape:", traj_s2.shape)
print("  min: %.4f, max: %.4f" % (traj_s2.min(), traj_s2.max()))

# src processing
print("\n--- SRC ---")
src_s1 = cv2.imread(src_path)
src_s1 = np.array(src_s1, dtype="float")
print("Step1: cv2.imread + dtype=float")
print("  min: %.2f, max: %.2f" % (src_s1.min(), src_s1.max()))

src_s2 = transform(np.array(src_s1, dtype="uint8")).float()
print("Step2: ToTensor() + float")
print("  shape:", src_s2.shape)
print("  min: %.4f, max: %.4f" % (src_s2.min(), src_s2.max()))

# label processing
print("\n--- LABEL ---")
label_s1 = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
print("Step1: cv2.imread(gray)")
print("  min: %d, max: %d" % (label_s1.min(), label_s1.max()))
print("  unique:", np.unique(label_s1))

label_s2 = np.expand_dims(label_s1, axis=-1)
print("Step2: expand_dims")
print("  shape:", label_s2.shape)

label_s3 = transform(label_s2).float()
print("Step3: ToTensor() + float (CORRECT)")
print("  min: %.4f, max: %.4f" % (label_s3.min(), label_s3.max()))
print("  unique:", np.unique(label_s3.numpy()))

label_s3_old = transform(label_s2).float() / 255.0
print("Step3: ToTensor()/255 (WRONG)")
print("  min: %.4f, max: %.4f" % (label_s3_old.min(), label_s3_old.max()))
print("  unique:", np.unique(label_s3_old.numpy()))

# building processing
print("\n--- BUILDING ---")
b_s1 = cv2.imread(building_path, cv2.IMREAD_GRAYSCALE)
b_s2 = np.expand_dims(b_s1, axis=-1)
b_s3 = transform(b_s2).float()
print("ToTensor() + float (CORRECT)")
print("  min: %.4f, max: %.4f" % (b_s3.min(), b_s3.max()))
print("  unique:", np.unique(b_s3.numpy()))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Raw data is uint8 [0,255]")
print("ToTensor() automatically converts to [0,1] float")
print("DO NOT divide by 255 again")
