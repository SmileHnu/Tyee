import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=50, output_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.relu = nn.ReLU()                        
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x