import torch.nn as nn

class DeepModel(nn.Module):

    def __init__(self, D_in, H=15, D_out=1):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        #print(x.size())
        x = self.fc2(x)
        #print(x.size())
        x = self.relu(x)
        #print(x.size())
        return x.squeeze()