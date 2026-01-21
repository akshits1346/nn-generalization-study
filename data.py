import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist_loaders(batch_size=128, data_fraction=1.0):
    """
    Returns train and test DataLoaders for MNIST.
    data_fraction allows controlled experiments with reduced dataset size.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    if data_fraction < 1.0:
        indices = np.random.permutation(len(train_dataset))
        subset_size = int(len(train_dataset) * data_fraction)
        train_dataset = Subset(train_dataset, indices[:subset_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

