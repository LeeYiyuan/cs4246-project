import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class RTrailNetwork(nn.Module):
    def __init__(self):
        super(RTrailNetwork, self).__init__()
        self.fc1 = nn.Linear(150, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3_output = nn.Linear(100, 50)
        self.fc3_hidden = nn.Linear(100, 50)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_output = torch.sigmoid(self.fc3_output(x))
        x_hidden = F.relu(self.fc3_hidden(x))
        return (x_output, x_hidden)

class ConvDQN(nn.Module):
    def __init__(self):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 2 * 42, 512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 2 * 42)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
