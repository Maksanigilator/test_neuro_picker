
import pybullet as p
import numpy as np
import torch
from torch.utils.data import Dataset
from environment import BinPickingEnv

class PickDataset(Dataset):
    def __init__(self, n_samples=50):
        self.env = BinPickingEnv(gui=False)
        self.data = []
        for i in range(n_samples):
            print(f"Generating sample {i+1}/{n_samples}")
            box_ids = self.env.reset_scene()
            feats, labels = [], []
            for bid in box_ids:
                x,y,z = p.getBasePositionAndOrientation(bid)[0]
                feats.append([z, x, y])
                ok = self.env.simulate_pick(bid)
                labels.append(float(ok))
            self.data.append((np.array(feats, dtype=np.float32), np.array(labels, dtype=np.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feats, labels = self.data[idx]
        return torch.from_numpy(feats), torch.from_numpy(labels)
