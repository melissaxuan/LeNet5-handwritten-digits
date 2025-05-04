from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import mnist
import torch
import numpy as np
import torchvision
from LeNet5 import LeNet5, ScaledTanh
 
def test(dataloader,model):

    #please implement your test code#
    ##HERE##
    ###########################                                                                                                                                                                               

    test_accuracy=0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch, (images, labels) in enumerate(dataloader):
            images = images / 255.0  # Normalize to [0.0, 1.0]

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            if (batch + 1) % 1000 == 0:
                print(f'Batch [{batch+1}/{len(dataloader)}] Test Accuracy: {accuracy:.2f}%')

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')


def main():
    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    mnist_test=mnist.MNIST(split="test",transform=pad)

    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("LeNet1.pth", weights_only=False)

    test(test_dataloader,model)

 

if __name__=="__main__":

    main()
