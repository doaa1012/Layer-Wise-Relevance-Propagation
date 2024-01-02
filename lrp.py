import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from utils import MNISTLoader
from model  import MNIST_CNN
logdir = './logs/'
chkpt = './logs/model.pth'
resultsdir = './results/'

class LayerwiseRelevancePropagation:
    def __init__(self, model):
        self.model = model
        self.epsilon = 1e-10

        # Load the model checkpoint
        self.model.load_state_dict(torch.load(chkpt))
        self.model.eval()  # Set the model to evaluation mode

        # Assuming self.model is a sequential model with named layers
        self.activations = []
        self.act_weights = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.Flatten, nn.Linear)):
                self.activations.append((name, module))

        self.relevances = self.get_relevances()

    def get_relevances(self):
        relevances = [self.activations[0][1], ]

        for i in range(1, len(self.activations)):
            name, layer = self.activations[i]
            if isinstance(layer, nn.Linear):
                relevances.append(self.backprop_fc(name, layer, relevances[-1]))
            elif isinstance(layer, nn.Flatten):
                relevances.append(self.backprop_flatten(layer, relevances[-1]))
            elif isinstance(layer, nn.MaxPool2d):
                relevances.append(self.backprop_max_pool2d(layer, relevances[-1]))
            elif isinstance(layer, nn.Conv2d):
                relevances.append(self.backprop_conv2d(name, layer, relevances[-1]))
            else:
                raise 'Error parsing layer!'

        return relevances

    def backprop_fc(self, name, activation, relevance):
   
        w = self.act_weights[name]
        w_pos = torch.clamp(w, min=0)  
        z = torch.matmul(activation, w_pos) + self.epsilon
        s = relevance / z
        c = torch.matmul(s, torch.transpose(w_pos, 0, 1))
        return c * activation

    def backprop_flatten(self, activation, relevance):
        shape = activation.size()  
        return relevance.view(shape)  

    def backprop_max_pool2d(self, activation, relevance, kernel_size=(2, 2), stride=(2, 2)):
        z = F.max_pool2d(activation, kernel_size, stride, padding='SAME') + self.epsilon
        s = relevance / z
        # Gradient calculation for max pooling is manual
        c = self._max_pool2d_grad(activation, z, s, kernel_size, stride)
        return c * activation

    def backprop_conv2d(self, name, activation, relevance, stride=(1, 1)):
        w = self.act_weights[name]
        w_pos = torch.clamp(w, min=0)
        z = F.conv2d(activation, w_pos, stride=stride, padding='SAME') + self.epsilon
        s = relevance / z
        # Calculate the gradients for conv2d manually
        c = F.conv2d(s, w_pos, stride=stride, padding='SAME')
        return c * activation

    #def _max_pool2d_grad(self, activation, z, s, kernel_size, stride):
        # Implement the gradient calculation for max pooling
        # This requires manual implementation and is specific to the architecture
        # This is a placeholder function
     #   pass


    def get_heatmap(self, digit):
        # Assuming MNISTLoader is a PyTorch DataLoader for MNIST dataset
        # Update this part as per your data loading mechanism
        for batch_idx, (data, target) in enumerate(self.dataloader):
            if target.item() == digit:
                sample = data
                break

        # Forward pass through the network to compute relevances
        # Note: Ensure that the model and data are on the same device (CPU/GPU)
        self.model.eval()  # Set the model to evaluation mode
        relevances = self.relevances[-1](sample)  # Assuming relevances[-1] is callable and returns the final relevance

        # Convert the tensor to numpy array and reshape
        heatmap = relevances.detach().numpy()[0].reshape(28, 28)
        heatmap /= heatmap.max()

        return heatmap

    def test(self):
        digit = np.random.choice(10)
        for batch_idx, (data, target) in enumerate(self.dataloader):
            if target.item() == digit:
                sample = data
                break

        # Forward pass through the network to compute relevances
        self.model.eval()  # Set the model to evaluation mode
        R = self.relevances(sample)  # Assuming relevances is callable and returns all relevances

        for r in R:
            print(r.sum().item())  # Assuming r is a PyTorch tensor

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = MNISTLoader()

    # Initialize model and LRP
    model = MNIST_CNN()
    lrp = LayerwiseRelevancePropagation(model)

    lrp.test()

    resultsdir = './results/'
    for digit in range(10):
        heatmap = lrp.get_heatmap(digit)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(heatmap, cmap='Reds', interpolation='bilinear')
        
        fig.savefig('{0}{1}.jpg'.format(resultsdir, digit))

