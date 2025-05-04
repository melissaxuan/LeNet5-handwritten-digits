from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import mnist
import torch
import numpy as np
from model2 import LeNet2

def test(dataloader, model):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images / 255.0  # Normalize manually
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print("Test accuracy:", test_accuracy)

def main():
    pad = transforms.Pad(2, fill=0, padding_mode='constant')
    mnist_test = mnist.MNIST(split="test", transform=pad)
    test_dataloader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    model = LeNet2()
    model.load_state_dict(torch.load("LeNet2.pth", map_location=torch.device("cpu")))
    model.eval()

    test(test_dataloader, model)

if __name__ == "__main__":
    main()
