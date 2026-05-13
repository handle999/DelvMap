import torch.utils.data as data
import numpy as np
import os
import json
import torchvision.transforms as transforms
import cv2


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float")
    return img


def get_data_loader_multistage(root_dir, mode):
    """
    Load dataset with train/val/test split support via split_indices.json
    mode: 'train', 'val', 'test'
    """
    dl = data.DataLoader(MultistageDataset(root_dir, mode), shuffle=(mode == 'train'), batch_size=1)
    return dl


class MultistageDataset(data.Dataset):
    def __init__(self, data_path, mode='train'):
        """
        data_path: dataset root directory (contains split_indices.json and data subdirs)
        mode: 'train', 'val', 'test'
        """
        super(MultistageDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode

        # Read split indices
        split_file = os.path.join(data_path, 'split_indices.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            self.selected_indices = split_data.get(mode, [])
            print(f"[{mode}] Loaded {len(self.selected_indices)} samples from split_indices.json")
        else:
            # Fallback to subdirectory mode
            data_path = os.path.join(data_path, mode)
            self.selected_indices = None
            print(f"[{mode}] No split_indices.json found, using subdirectory: {data_path}")

        # Get data directories
        traj_dir = os.path.join(data_path if self.selected_indices is None else data_path, 'traj_and_point_split')

        # Get all file names (without extension)
        all_traj_files = sorted([f.replace('.npy', '') for f in os.listdir(traj_dir) if f.endswith('.npy')])

        if self.selected_indices is not None:
            # Use indices from split_indices.json
            self.file_indices = [all_traj_files[i] for i in self.selected_indices]
        else:
            # Use all files
            self.file_indices = all_traj_files

    def __getitem__(self, index):
        file_idx = self.file_indices[index]

        traj_path = os.path.join(self.data_path, 'traj_and_point_split', f'{file_idx}.npy')
        label_path = os.path.join(self.data_path, 'label', f'{file_idx}.png')
        src_path = os.path.join(self.data_path, 'src_split', f'{file_idx}.png')
        building_label_path = os.path.join(self.data_path, 'building_label', f'{file_idx}.png')
        img_path = traj_path

        traj = np.asarray(np.load(traj_path))
        traj = np.array(traj, dtype="float")
        label = load_img(str(label_path), grayscale=True)
        label = np.expand_dims(label, axis=-1)

        src = load_img(src_path, grayscale=False)
        building_label = load_img(str(building_label_path), grayscale=True)
        building_label = np.expand_dims(building_label, axis=-1)

        img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        img = img_transform(np.array(traj, dtype="uint8")).float()
        label = img_transform(label * 128).float()
        src = img_transform(np.array(src, dtype="uint8")).float()
        building_label = img_transform(building_label * 255).float()

        return {
            'traj_path': traj_path,
            'traj_data': img,
            'label_path': label_path,
            'label_data': label,
            'src_path': src_path,
            'src_data': src,
            'building_label_path': building_label_path,
            'building_label_data': building_label,
            'img_path': img_path,
        }

    def __len__(self):
        return len(self.file_indices)
