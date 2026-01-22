import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_rotated_mnist_loader(angle, batch_size=128):
    transform = transforms.Compose([
        transforms.RandomRotation((angle, angle)),
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


