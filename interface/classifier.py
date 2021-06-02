import torch
import torch.nn as nn


class Classifier(nn.Module):
    """ input shape: (-1, 3, 32, 32) """

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential()
        self.seq.add_module('conv-1', nn.Conv2d(3, 32, (5, 5), 1, 2))
        self.seq.add_module('relu-1', nn.ReLU())
        self.seq.add_module('pool-1', nn.MaxPool2d(3, 2))
        self.seq.add_module('conv-2', nn.Conv2d(32, 64, (3, 3), 1, 1))
        self.seq.add_module('relu-2', nn.ReLU())
        self.seq.add_module('pool-2', nn.MaxPool2d(3, 2))
        self.seq.add_module('conv-3', nn.Conv2d(64, 96, (3, 3), 1, 1))
        self.seq.add_module('relu-3', nn.ReLU())
        self.seq.add_module('conv-4', nn.Conv2d(96, 96, (3, 3), 1, 1))
        self.seq.add_module('relu-4', nn.ReLU())
        self.seq.add_module('conv-5', nn.Conv2d(96, 64, (3, 3), 1, 1))
        self.seq.add_module('relu-5', nn.ReLU())
        self.seq.add_module('pool-5', nn.MaxPool2d(3, 2))

        self.seq.add_module('flat', nn.Flatten())

        self.seq.add_module('drop-1', nn.Dropout())
        self.seq.add_module('lin-1', nn.Linear(576, 256))
        self.seq.add_module('lin-relu-1', nn.ReLU())
        self.seq.add_module('drop-2', nn.Dropout())
        self.seq.add_module('lin-2', nn.Linear(256, 64))
        self.seq.add_module('lin-relu-2', nn.ReLU())
        self.seq.add_module('lin-3', nn.Linear(64, 2))

    def forward(self, x):
        return self.seq(x)


def load_classifier(path):
    return torch.load(path, map_location=torch.device('cpu'))
