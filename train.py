import torch
import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch.
    Returns average loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluates the model without gradient computation.
    Returns average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy

