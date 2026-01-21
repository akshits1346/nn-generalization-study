import torch
import torch.optim as optim

from data import get_mnist_loaders
from models import LogisticRegression, MLP
from train import train_one_epoch, evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_experiment(model, data_fraction=1.0, epochs=10, lr=1e-3):
    train_loader, test_loader = get_mnist_loaders(data_fraction=data_fraction)

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, DEVICE
        )
        val_loss, val_acc = evaluate(
            model, test_loader, DEVICE
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(
            f"Epoch {epoch}: "
            f"Train Acc={train_acc:.3f}, "
            f"Val Acc={val_acc:.3f}"
        )

    return history


if __name__ == "__main__":
    model = MLP(depth=2)
    run_experiment(model)

