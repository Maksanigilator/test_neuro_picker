import pybullet as p  # Для доступа к p.getBasePositionAndOrientation и т.д.
import numpy as np
import torch
from torch.utils.data import Dataset
from environment_simple import BinPickingEnv
from scipy.ndimage import zoom
from multiprocessing import Pool, cpu_count


class PickDataset(Dataset):
    def __init__(self, n_samples=100, crop_size=64, visualize=False, num_processes=None):
        self.crop_size = crop_size
        self.visualize = visualize
        self.num_processes = num_processes or cpu_count()
        self.data = self.generate_dataset(n_samples)

    def generate_dataset(self, n_samples):
        # Распараллеливаем генерацию сэмплов
        with Pool(self.num_processes) as pool:
            results = pool.map(self.generate_single_sample, range(n_samples))

        data = []
        for sample_data in results:
            data.extend(sample_data)

        return data

    def generate_single_sample(self, i):
        # Каждый процесс создаёт свой env
        env = BinPickingEnv(gui=self.visualize)
        if not self.visualize:
            print(f"Generating sample {i + 1}")

        box_ids = env.reset_scene()
        rs_data = env.get_realsense_data()
        depth_img = rs_data['depth']
        seg_mask = rs_data['seg']

        # Сохраняем исходные состояния всех коробок
        initial_state = {}
        for bid in box_ids:
            pos, orn = p.getBasePositionAndOrientation(bid)
            lin_vel, ang_vel = p.getBaseVelocity(bid)
            initial_state[bid] = (pos, orn, lin_vel, ang_vel)

        unique_ids = np.unique(seg_mask)
        sample_data = []
        for uid in unique_ids:
            if uid == -1 or uid not in box_ids:
                continue
            obj_mask = (seg_mask == uid)
            if np.sum(obj_mask) < 10: continue

            v_coords, u_coords = np.nonzero(obj_mask)
            center_v, center_u = int(np.mean(v_coords)), int(np.mean(u_coords))

            half = self.crop_size // 2
            depth_crop = self._crop_image(depth_img, center_v, center_u, half)
            mask_crop = self._crop_image(obj_mask.astype(np.float32), center_v, center_u, half)

            input_crop = np.stack([depth_crop, mask_crop], axis=0)  # 2 x 64 x 64

            # Восстанавливаем исходные состояния
            for bid in box_ids:
                pos, orn, lin_vel, ang_vel = initial_state[bid]
                p.resetBasePositionAndOrientation(bid, pos, orn)
                p.resetBaseVelocity(bid, lin_vel, ang_vel)

            success = env.simulate_pick(uid, remove_on_fail=False)
            label = 1.0 if success else 0.0

            sample_data.append((input_crop, label))

        env.close()  # Закрываем env в процессе
        return sample_data

    def _crop_image(self, img, center_v, center_u, half):
        h, w = img.shape
        top = max(0, center_v - half)
        bottom = min(h, center_v + half)
        left = max(0, center_u - half)
        right = min(w, center_u + half)
        crop = img[top:bottom, left:right]
        pad_top = half - (center_v - top)
        pad_bottom = half - (bottom - center_v)
        pad_left = half - (center_u - left)
        pad_right = half - (right - center_u)
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        if crop.shape != (self.crop_size, self.crop_size):
            crop = zoom(crop, (self.crop_size / crop.shape[0], self.crop_size / crop.shape[1]))
        return crop.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        crop, label = self.data[idx]
        return torch.from_numpy(crop), torch.tensor(label).float()