# Weighted hinge loss function
# input: specify the weight of class 1, with respect to class 0
# the idea is that the weight is 1 for instances of class 0, and w_1_class for instances of class 1
# inspired from: https://stackoverflow.com/questions/55754976/weighted-hinge-loss-function
import torch
# def weightedSquaredHingeLoss(inp, tar, w_1_class=1):
#     device = inp.device
#     return torch.sum(torch.mean((torch.max(tar, torch.zeros(inp.shape[1], dtype=torch.float32).to(device))*(w_1_class-1)+1) * torch.max(1. - tar * inp, torch.zeros(inp.shape[1], dtype=torch.float32).to(device))**2, dim=-1))

class weightedSquaredHingeLoss(torch.nn.Module):
    def __init__(self):
        super(weightedSquaredHingeLoss, self).__init__()
    
    def forward(self, inp, tar, w_1_class=1):
        """
        inp: (batch_size, num_classes)
        tar: (batch_size, num_classes)
        """
        device = inp.device
        return torch.sum(torch.mean((torch.max(tar, torch.zeros(inp.shape[1], dtype=torch.float32).to(device))*(w_1_class-1)+1) * torch.max(1. - tar * inp, torch.zeros(inp.shape[1], dtype=torch.float32).to(device))**2, dim=-1))
