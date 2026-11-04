import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dims):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

    