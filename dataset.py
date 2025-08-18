import pybullet as p  # Для доступа к p.getBasePositionAndOrientation и т.д.
import numpy as np
import torch
from torch.utils.data import Dataset
from environment import BinPickingEnv
from scipy.ndimage import zoom

class PickDataset(Dataset):
    def __init__(self, n_samples=100, crop_size=64):
        self.env = BinPickingEnv(gui=False)
        self.crop_size = crop_size
        self.data = self.generate_dataset(n_samples)

    def generate_dataset(self, n_samples):
        data = []
        for i in range(n_samples):
            print(f"Generating sample {i + 1}/{n_samples}")
            box_ids = self.env.reset_scene()
            rs_data = self.env.get_realsense_data()
            depth_img = rs_data['depth']
            seg_mask = rs_data['seg']

            # Сохраняем исходные состояния всех коробок (pos, orn, lin_vel, ang_vel)
            initial_state = {}
            for bid in box_ids:
                pos, orn = p.getBasePositionAndOrientation(bid)
                lin_vel, ang_vel = p.getBaseVelocity(bid)
                initial_state[bid] = (pos, orn, lin_vel, ang_vel)

            unique_ids = np.unique(seg_mask)
            for uid in unique_ids:
                if uid == -1 or uid not in box_ids:  # Пропускаем фон, plane, pallet
                    continue
                obj_mask = (seg_mask == uid)
                if np.sum(obj_mask) < 10: continue  # Слишком маленький объект

                # Центр маски (v=height, u=width)
                v_coords, u_coords = np.nonzero(obj_mask)
                center_v, center_u = int(np.mean(v_coords)), int(np.mean(u_coords))

                # Crop depth и mask
                half = self.crop_size // 2
                depth_crop = self._crop_image(depth_img, center_v, center_u, half)
                mask_crop = self._crop_image(obj_mask.astype(np.float32), center_v, center_u, half)

                # Stack to 2-channel
                input_crop = np.stack([depth_crop, mask_crop], axis=0)  # 2 x 64 x 64

                # Восстанавливаем исходные состояния всех коробок
                for bid in box_ids:
                    pos, orn, lin_vel, ang_vel = initial_state[bid]
                    p.resetBasePositionAndOrientation(bid, pos, orn)
                    p.resetBaseVelocity(bid, lin_vel, ang_vel)

                # Симулируем захват без удаления (для обучения)
                success = self.env.simulate_pick(uid, remove_on_fail=False)
                label = 1.0 if success else 0.0

                data.append((input_crop, label))

        return data

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