import torch
import torch.nn as nn
import torch.optim as optim
from model_utils import BrainTumorModel, BrainMRIDataset
from torch.utils.data import DataLoader

# Parameters
epochs = 10
batch_size = 8
learning_rate = 0.001

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoader
dataset = BrainMRIDataset("Brain_Tumor", height=128, width=128)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss, optimizer
model = BrainTumorModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# Save trained model
torch.save(model.state_dict(), "model.pth")
print("✅ model.pth has been saved!")
