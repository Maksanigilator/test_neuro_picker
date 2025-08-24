# generate_datasets.py (in src/)
import pybullet as p
import numpy as np
import h5py
import yaml
import os
import signal
import sys
from multiprocessing import Pool, cpu_count
from environment import BinPickingEnv
import random  # For random split decision


class DatasetGenerator:
    def __init__(self, config_path='../config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.train_dir = self.config['dataset']['train_dir']  # e.g., '../datasets/train/'
        self.val_dir = self.config['dataset']['val_dir']  # e.g., '../datasets/val/'
        self.n_scenes = self.config['dataset']['n_scenes']  # Scenes to generate in this run
        self.split_ratio = 0.8  # 80/20 chance per scene
        self.num_processes = cpu_count()

        if not os.path.exists(self.train_dir) or not os.path.exists(self.val_dir):
            raise FileNotFoundError(f"Directories not found: {self.train_dir} or {self.val_dir}. Create them manually.")

        # Get existing counts
        self.train_count = len([f for f in os.listdir(self.train_dir) if f.endswith('.h5')])
        self.val_count = len([f for f in os.listdir(self.val_dir) if f.endswith('.h5')])

        # Signal handler for Ctrl+C (but since we save after each, just print)
        signal.signal(signal.SIGINT, self._interrupt_handler)

    def _interrupt_handler(self, sig, frame):
        print("Interrupted! All generated scenes up to now are saved.")
        sys.exit(0)

    def generate_datasets(self):
        args = range(self.n_scenes)  # Indices for scenes
        with Pool(self.num_processes) as pool:
            results = pool.imap(self.generate_single_scene, args)  # Ordered iterator
            for i, scene_data in enumerate(results):
                self._save_scene(scene_data, i)

    def generate_single_scene(self, i):
        print(f"Generating scene {i + 1}")
        env = BinPickingEnv(gui=False)

        box_ids = env.reset_scene()
        rs_data = env.get_realsense_data()
        rgb_img = rs_data['rgb']
        depth_img = rs_data['depth']
        seg_mask = rs_data['seg']

        initial_state = {}
        for bid in box_ids:
            pos, orn = p.getBasePositionAndOrientation(bid)
            lin_vel, ang_vel = p.getBaseVelocity(bid)
            initial_state[bid] = (pos, orn, lin_vel, ang_vel)

        unique_ids = np.unique(seg_mask)
        objects_data = {}
        for uid in unique_ids:
            if uid == -1 or uid not in box_ids:
                continue
            obj_mask = (seg_mask == uid)
            if np.sum(obj_mask) < 10: continue

            v_coords, u_coords = np.nonzero(obj_mask)
            center_v, center_u = int(np.mean(v_coords)), int(np.mean(u_coords))
            z = depth_img[center_v, center_u]

            for bid in box_ids:
                pos, orn, lin_vel, ang_vel = initial_state[bid]
                p.resetBasePositionAndOrientation(bid, pos, orn)
                p.resetBaseVelocity(bid, lin_vel, ang_vel)

            success = env.simulate_pick(uid, remove_on_fail=False)
            label = 1.0 if success else 0.0

            objects_data[uid] = {'label': label, 'center_u': center_u, 'center_v': center_v, 'z': z}

        scene_data = {
            'rgb': rgb_img,
            'depth': depth_img,
            'seg': seg_mask,
            'objects': objects_data
        }

        env.close()
        return scene_data

    def _save_scene(self, scene_data, i):
        # Decide split: random with 80% chance train
        is_train = random.random() < self.split_ratio
        split = 'train' if is_train else 'val'
        count = self.train_count if is_train else self.val_count
        dir_path = self.train_dir if is_train else self.val_dir

        file_path = os.path.join(dir_path, f'scene_{count + 1}.h5')
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rgb', data=scene_data['rgb'])
            hf.create_dataset('depth', data=scene_data['depth'])
            hf.create_dataset('seg', data=scene_data['seg'])
            obj_group = hf.create_group('objects')
            for uid, obj_data in scene_data['objects'].items():
                sub_group = obj_group.create_group(str(uid))
                sub_group.attrs['label'] = obj_data['label']
                sub_group.attrs['center_u'] = obj_data['center_u']
                sub_group.attrs['center_v'] = obj_data['center_v']
                sub_group.attrs['z'] = obj_data['z']

        print(f"Generated {split} scene {count + 1}")

        # Update counts
        if is_train:
            self.train_count += 1
        else:
            self.val_count += 1


if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate_datasets()