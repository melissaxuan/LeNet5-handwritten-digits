import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from model2 import LeNet2
import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet2().to(device)

# Use only padding (no ToTensor or Normalize because input is already a tensor)
pad = transforms.Pad(2, fill=0, padding_mode='constant')

# Load dataset
trainset = mnist.MNIST(split="train", transform=pad)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(20):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs = inputs / 255.0  # Normalize manually to [0.0, 1.0]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

torch.save(model.state_dict(), "LeNet2.pth")
