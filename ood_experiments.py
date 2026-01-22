import torch
import torch.optim as optim

from models import MLP
from train import train_one_epoch, evaluate
from data import get_mnist_loaders
from ood_data import get_rotated_mnist_loader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_and_evaluate_ood(angle):
    train_loader, test_loader = get_mnist_loaders()
    ood_loader = get_rotated_mnist_loader(angle)

    model = MLP(depth=2, dropout=0.5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(10):
        train_one_epoch(model, train_loader, optimizer, DEVICE)

    id_loss, id_acc = evaluate(model, test_loader, DEVICE)
    ood_loss, ood_acc = evaluate(model, ood_loader, DEVICE)

    print(f"Rotation {angle}° → ID Acc: {id_acc:.3f}, OOD Acc: {ood_acc:.3f}")


