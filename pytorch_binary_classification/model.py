import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.dense1 = nn.Linear(33, 62)
        self.dense2 = nn.Linear(62,62)
        self.dense3 = nn.Linear(62, 128)
        self.dense4 = nn.Linear(128, 128)
        self.dense6 = nn.Linear(128, 1)


    def forward(self, input):
        output = input.view(-1,33)
        output = F.relu(self.dense1(output))
        output = F.relu(self.dense2(output))
        output = F.relu(self.dense3(output))
        output = F.relu(self.dense4(output))
        output = self.dense6(output)


        return output