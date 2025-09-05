# apply MNIST dataset 
import torch
from torchvision import datasets, transforms

def get_mnist_dataloader(batch_size):
    mnist = datasets.MNIST(root='.././data', train=True, download=True, transform=transforms.ToTensor())
    x, y = mnist.data, mnist.targets
    x = x.reshape(-1, 28, 28, 1).float() / 255.0
    y = y.long()
    train_dataset = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return data_loader