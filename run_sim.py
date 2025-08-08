
import pybullet as p
import torch
import numpy as np
import time
from environment import BinPickingEnv
from model import PickerNet

# Подготовка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PickerNet().to(device)
try:
    model.load_state_dict(torch.load("picker_net.pth", map_location=device))
    print("Loaded picker_net.pth")
except FileNotFoundError:
    model.load_state_dict(torch.load("picker_net_interrupted.pth", map_location=device))
    print("Loaded picker_net_interrupted.pth")
model.eval()

# Запуск GUI и разбор кучи
env = BinPickingEnv(gui=True)
box_ids = env.reset_scene()

# Цикл разбора одной за одной
while box_ids:
    feats = []
    for bid in box_ids:
        x, y, z = p.getBasePositionAndOrientation(bid)[0]
        feats.append([z, x, y])
    feats = np.array(feats, dtype=np.float32)

    # Предсказание и выбор лучшего
    with torch.no_grad():
        scores = model(torch.from_numpy(feats).unsqueeze(0).to(device))
        scores = scores.cpu().numpy().reshape(-1)
    best_idx = int(np.argmax(scores))
    chosen = box_ids[best_idx]
    print(f"Picking object {chosen} (score {scores[best_idx]:.3f}), remaining {len(box_ids)} boxes")

    # Симуляция захвата
    success = env.simulate_pick(chosen)
    print(" -> success" if success else " -> failed")

    # Удаляем поднятую коробку
    env.remove_box(chosen)
    # Удаляем упавшие коробки после захвата
    env.remove_fallen(env.box_ids)
    # Обновляем список оставшихся
    box_ids = env.box_ids.copy()
    box_ids = env.box_ids.copy()

    # Ждём стабилизации
    for _ in range(int(240 * 0.5)):
        p.stepSimulation()
        time.sleep(env.time_step)

print("All boxes picked")
env.close()
