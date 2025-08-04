
import pybullet as p
import torch
import numpy as np
from environment import BinPickingEnv
from model import PickerNet

# Устройство и загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PickerNet().to(device)
loaded = False
for fname in ("picker_net.pth","picker_net_interrupted.pth"):
    try:
        model.load_state_dict(torch.load(fname, map_location=device))
        print(f"Loaded {fname}")
        loaded = True
        break
    except FileNotFoundError:
        continue
if not loaded:
    raise FileNotFoundError("Нет сохранённой модели. Сначала запустите train.py")
model.eval()

# GUI-сессия
env = BinPickingEnv(gui=True)
while True:
    box_ids = env.reset_scene()
    feats = []
    for bid in box_ids:
        x,y,z = p.getBasePositionAndOrientation(bid)[0]
        feats.append([z,x,y])
    feats = np.array(feats, dtype=np.float32)
    # предсказание
    with torch.no_grad():
        scores = model(torch.from_numpy(feats).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
    best = int(np.argmax(scores))
    chosen = box_ids[best]
    print(f"Picking {chosen}, score {scores[best]:.3f}")
    ok = env.simulate_pick(chosen)
    print("Success" if ok else "Failed")
    cmd = input("Enter=next, q=quit:")
    if cmd.lower().startswith('q'):
        break

env.close()
