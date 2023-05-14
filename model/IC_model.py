import torch
import torch.nn as nn
import math

class ICModel(nn.Module):
    def __init__(self, input_size, out_size):
        super(ICModel, self).__init__()

        self.out_size = out_size
        self.l2 = torch.nn.Linear(input_size, out_size)
        self.linear = torch.nn.Linear(2*out_size, out_size)

    def forward(self, x):
        x = x.float()
        x1 = torch.sum(x, 1).unsqueeze(1)   # [batch, 1]
        x1 = x1.repeat(1, self.out_size)    # [batch, out_size]
        x2 = self.l2(x)                     # [batch, out_size]
        x = torch.cat((x1, x2), 1)          # [batch, out_size*2]
        x = self.linear(x)                  # [batch, out_size]
        x = torch.relu(x)
        x = x2 + x
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(2)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ICFC(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(ICFC, self).__init__()
        self.hidden_size = hidden_size
        self.IC1 = ICModel(input_size, hidden_size)
        self.IC2 = ICModel(hidden_size, out_size)
    
    def forward(self, x):
        x = self.IC1(x)
        x = self.IC2(x)
        return x 

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


model = ICModel(10,2)
i = torch.zeros(64, 10)
o = model(i)
print(o.shape)