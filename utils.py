import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTLoader:
    def __init__(self, batch_size=100, download_path='./data'):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_dataset = datasets.MNIST(download_path, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(download_path, train=False, download=True, transform=self.transform)

    def get_loader(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

    @property
    def train(self):
        return self.train_dataset

    @property
    def test(self):
        return self.test_dataset

if __name__ == '__main__':
    dl = MNISTLoader()

    train_loader = dl.get_loader(train=True)
    test_loader = dl.get_loader(train=False)

    # Example of iterating through data
    for i, (x, y) in enumerate(train_loader):
        print(i, x.shape, y.shape)
        if i == 4:  # Just for demonstration
            break

    # Accessing the dataset directly
    print('Train dataset size:', len(dl.train))
    print('Test dataset size:', len(dl.test))

