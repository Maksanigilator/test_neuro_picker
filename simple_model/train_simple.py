import torch
from torch.utils.data import DataLoader
from dataset_simple import PickDataset
from model_simple import PickerNet
import torch.optim as optim
import signal
import sys
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

n_samples = 2000  # Можно увеличить
epochs = 200
crop_size = 64

dataset = PickDataset(n_samples=n_samples, crop_size=crop_size)
loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Batch >1 для CUDA

model = PickerNet(crop_size=crop_size).to(device)
# Загружаем существующие веса, если файл существует (для дообучения)
try:
    model.load_state_dict(torch.load("picker_net(small).pth", map_location=device))
    print("Loaded existing picker_net(small).pth for fine-tuning")
except FileNotFoundError:
    print("No existing model found, starting from scratch")

opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

def handler(sig, frame):
    print("Interrupted, saving...")
    torch.save(model.state_dict(), "picker_net_interrupted.pth")
    sys.exit(0)
signal.signal(signal.SIGINT, handler)

losses = []  # Для графика
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
    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch}/{epochs}, loss={avg_loss:.4f}")

torch.save(model.state_dict(), "picker_net(small).pth")  # Перезаписываем существующий файл
print("Training finished.")

# Строим и сохраняем график
plt.figure()
plt.plot(range(1, epochs+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()
print("Training loss graph saved as training_loss.png")