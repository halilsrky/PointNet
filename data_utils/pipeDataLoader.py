def rotate_point_cloud_z(batch_data):
    """ Z ekseninde rastgele döndürme (augmentasyon için).
    batch_data: [B, N, 3] veya [B, N, C]
    """
    B, N, C = batch_data.shape
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(B):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval,  cosval, 0],
                                    [0,       0,      1]])
        rotated_data[k, :, :3] = np.dot(batch_data[k, :, :3], rotation_matrix)
        if C > 3:
            rotated_data[k, :, 3:] = batch_data[k, :, 3:]
    return rotated_data
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class CustomPointDataset(Dataset):
    def __init__(self, split='train', data_root='NPY', num_point=4096, test_range=(241, 300), block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        # Dosyaları sırala (augmentli dosyalar dahil)
        files = sorted([f for f in os.listdir(data_root) if f.endswith('.npy') and f.startswith('TestRay')])

        # Augmentli dosyaları da dahil et (sadece eğitimde)
        if split == 'train':
            files_split = [f for f in files if int(''.join(filter(str.isdigit, f))) < test_range[0]]
            # augmentli dosyalar: *_rot.npy, *_scale.npy, *_trans.npy
            aug_files = []
            for f in files_split:
                base = f.replace('.npy', '')
                for aug in ['_rot.npy', '_scale.npy', '_trans.npy']:
                    aug_file = base + aug
                    if aug_file in files:
                        aug_files.append(aug_file)
            files_split += aug_files
        else:
            files_split = [f for f in files if int(''.join(filter(str.isdigit, f))) >= test_range[0] and int(''.join(filter(str.isdigit, f))) <= test_range[1]]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        
        # Kaç sınıf varsa ona göre labelweights boyutunu değiştir
        num_classes = 2  # Örnek, ihtiyacına göre değiştir
        labelweights = np.zeros(num_classes)

        for file_name in tqdm(files_split, total=len(files_split)):
            file_path = os.path.join(data_root, file_name)
            data = np.load(file_path)  # Nx4: X Y Z label
            points = data[:, 0:3]      # X,Y,Z
            labels = data[:, 3].astype(int)

            tmp, _ = np.histogram(labels, bins=np.arange(num_classes+1))
            labelweights += tmp

            coord_min, coord_max = np.min(points, axis=0), np.max(points, axis=0)
            self.room_points.append(points)
            self.room_labels.append(labels)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.max(labelweights) / (labelweights + 1e-6), 1/3.0)
        print('Label weights:', self.labelweights)

        sample_prob = np.array(num_point_all) / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for idx in range(len(files_split)):
            room_idxs.extend([idx] * int(round(sample_prob[idx] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print(f'Totally {len(self.room_idxs)} samples in {split} set.')

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N x 3
        labels = self.room_labels[room_idx]  # N
        N_points = points.shape[0]

        max_try = 100
        for try_idx in range(max_try):
            center = points[np.random.choice(N_points)]
            block_min = center[:2] - self.block_size / 2.0
            block_max = center[:2] + self.block_size / 2.0

            # XY içinde kalan noktaları bul
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
                (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
            )[0]

            if point_idxs.size > 1024:
                break
        else:
            # 100 denemede de yeterli nokta bulunamazsa, tüm noktaları kullan
            point_idxs = np.arange(N_points)

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        selected_points = points[selected_point_idxs, :]  # num_point x 3
        current_labels = labels[selected_point_idxs]

        # Blok merkezine göre normalize et (S3DIS mantığına benzer)
        # Yani, blok merkezini (center) sıfır noktası yap
        selected_points = selected_points - center[:3]

        # İstersen xyz'yi [0,1] aralığına çekmek için aşağıdaki satırı açabilirsin:
        # selected_points = (selected_points - self.room_coord_min[room_idx]) / (self.room_coord_max[room_idx] - self.room_coord_min[room_idx] + 1e-6)

        if self.transform is not None:
            selected_points, current_labels = self.transform(selected_points, current_labels)

        # Model girişine uygun: [num_point, 3] ve [num_point]
        return selected_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

