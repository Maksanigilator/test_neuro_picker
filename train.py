import torch
from torch.utils.data import DataLoader
from dataset import PickDataset
from model import PickerNet
import torch.optim as optim
import signal, sys

# Проверка CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Параметры обучения
n_samples = 7
epochs = 5

# Датасет и загрузчик
from torch.utils.data import DataLoader
dataset = PickDataset(n_samples=n_samples)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Модель и оптимизатор
model = PickerNet().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

# Обработчик Ctrl+C для сохранения
def handler(sig, frame):
    print("Interrupted, saving...")
    torch.save(model.state_dict(), "picker_net_interrupted.pth")
    sys.exit(0)
signal.signal(signal.SIGINT, handler)
# Обучение
for epoch in range(1, epochs+1):
    total_loss = 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        preds = model(feats)
        loss = criterion(preds, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{epochs}, loss={total_loss/len(loader):.4f}")
# Сохраняем модель
torch.save(model.state_dict(), "picker_net.pth")
print("Training finished.")