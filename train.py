import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import MNISTLoader
from model import MNIST_CNN
# MNIST_CNN definition (assuming this is a PyTorch compatible class)

class Trainer:
    def __init__(self, model, batch_size=100, n_epochs=1, learning_rate=0.001):
        self.model = model
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # MNIST datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.validation_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def run(self):
        for epoch in range(self.n_epochs):
            self.train(epoch)
            self.validate()

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        for data, target in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_accuracy = total_correct / len(self.train_loader.dataset)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')

    def validate(self):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()

        avg_accuracy = total_correct / len(self.validation_loader.dataset)
        print(f'Validation Accuracy: {avg_accuracy:.4f}')

if __name__ == '__main__':
    model = MNIST_CNN()  
    trainer = Trainer(model)
    trainer.run()
