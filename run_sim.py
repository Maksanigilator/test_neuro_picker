import pybullet as p
import torch
import numpy as np
import time
from environment import BinPickingEnv
from model import PickerNet
from scipy.ndimage import zoom  # Для crop

def _crop_image(img, center_v, center_u, half):
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
    if crop.shape != (crop_size, crop_size):
        crop = zoom(crop, (crop_size / crop.shape[0], crop_size / crop.shape[1]))
    return crop.astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crop_size = 64

# Load model
model = PickerNet(crop_size=crop_size).to(device)
try:
    model.load_state_dict(torch.load("picker_net.pth", map_location=device))
except FileNotFoundError:
    model.load_state_dict(torch.load("picker_net_interrupted.pth", map_location=device))
model.eval()

# Запуск
env = BinPickingEnv(gui=True)
box_ids = env.reset_scene()

while box_ids:
    rs_data = env.get_realsense_data()
    depth_img = rs_data['depth']
    seg_mask = rs_data['seg']

    inputs = []
    obj_ids = []
    unique_ids = np.unique(seg_mask)
    for uid in unique_ids:
        if uid == -1 or uid not in box_ids:  # Пропускаем фон, plane, pallet
            continue
        obj_mask = (seg_mask == uid)
        if np.sum(obj_mask) < 10: continue

        v_coords, u_coords = np.nonzero(obj_mask)
        center_v, center_u = int(np.mean(v_coords)), int(np.mean(u_coords))

        half = crop_size // 2
        depth_crop = _crop_image(depth_img, center_v, center_u, half)
        mask_crop = _crop_image(obj_mask.astype(np.float32), center_v, center_u, half)
        input_crop = np.stack([depth_crop, mask_crop], axis=0)  # 2 x 64 x 64

        inputs.append(input_crop)
        obj_ids.append(uid)

    if not inputs:
        break

    inputs_tensor = torch.from_numpy(np.stack(inputs)).to(device)  # batch x 2 x 64 x 64
    with torch.no_grad():
        logits = model(inputs_tensor)
        scores = torch.sigmoid(logits).cpu().numpy().flatten()

    best_idx = np.argmax(scores)
    chosen = obj_ids[best_idx]
    print(f"Picking object {chosen} (score {scores[best_idx]:.3f}), remaining {len(box_ids)} boxes")

    success = env.simulate_pick(chosen)
    print(" -> success" if success else " -> failed")

    env.remove_box(chosen)
    env.remove_fallen(env.box_ids)
    box_ids = env.box_ids.copy()

    for _ in range(int(240 * 0.5)):
        p.stepSimulation()
        time.sleep(env.time_step)

print("All boxes picked")
env.close()
