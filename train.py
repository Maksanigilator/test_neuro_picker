import torch
from torch.utils.data import DataLoader
from dataset import PickDataset
from model import PickerNet
import torch.optim as optim
import signal
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

n_samples = 100  # Можно увеличить
epochs = 80
crop_size = 64

dataset = PickDataset(n_samples=n_samples, crop_size=crop_size)
loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Batch >1 для CUDA

model = PickerNet(crop_size=crop_size).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

def handler(sig, frame):
    print("Interrupted, saving...")
    torch.save(model.state_dict(), "picker_net_interrupted.pth")
    sys.exit(0)
signal.signal(signal.SIGINT, handler)

for epoch in range(1, epochs+1):
    total_loss = 0
    for crops, labels in loader:
        crops, labels = crops.to(device), labels.to(device).unsqueeze(1)  # Labels to [batch,1]
        preds = model(crops)
        loss = criterion(preds, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{epochs}, loss={total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "picker_net.pth")
print("Training finished.")