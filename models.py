import torch.nn as nn

class LogisticRegression(nn.Module):
    """
    Baseline linear classifier for MNIST.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable depth.
    Used to study capacity and overfitting.
    """
    def __init__(self, depth=2, hidden_dim=256, dropout=0.0):
        super().__init__()
        layers = []
        input_dim = 28 * 28

        for _ in range(depth):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


