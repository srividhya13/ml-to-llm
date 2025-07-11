import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear_layer=nn.Linear(784,512)
        self.second_layer=nn.ReLU()
        self.dropout=nn.Dropout(p=0.2)
        self.final_layer=nn.Linear(512,10)
        self.sigmoid=nn.Sigmoid()
        
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        op=self.sigmoid(self.final_layer(self.dropout(self.second_layer(self.linear_layer(images)))))
        return torch.round(op,decimals=4)
