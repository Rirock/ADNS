import math
import torch
from torch import nn
import torch.nn.functional as F


class ADNM(nn.Module):
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(ADNM, self).__init__()

        self.input_size = input_size
        w = torch.rand([out_size, M, input_size]).to(device)
        q = torch.rand([out_size, M, input_size]).to(device)
        m = torch.rand([out_size, 1, input_size]).to(device)
        torch.nn.init.constant_(q, 0.1)
        torch.nn.init.uniform_(m, a=-10.0, b=10.0)
        
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'m': nn.Parameter(m)})

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['w'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        S = torch.sigmoid(torch.mul(x, self.params['w']) - self.params['q'])

        A = torch.tanh(self.params['m'])       
        # Dendritic
        D = torch.sum(torch.mul(S, A), 3) 

        # Membrane Soma
        O = torch.sum(torch.sigmoid(D), 2)

        return O
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class ADNS(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(ADNS, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = ADNM(input_size, hidden_size, M)
        self.DNM_Linear2 = ADNM(hidden_size, out_size, M)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        out = self.DNM_Linear2(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)



class DNM_Linear(nn.Module):    # MDNN
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(DNM_Linear, self).__init__()

        Synapse_W = torch.rand([out_size, M, input_size]).to(device)#.cuda() # [size_out, M, size_in]
        Synapse_q = torch.rand([out_size, M, input_size]).to(device)#.cuda()
        torch.nn.init.constant_(Synapse_q, 0.1)
        k = torch.rand(1).to(device)
        qs = torch.rand(1).to(device)

        self.params = nn.ParameterDict({'Synapse_W': nn.Parameter(Synapse_W)})
        self.params.update({'Synapse_q': nn.Parameter(Synapse_q)})
        self.params.update({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})
        self.input_size = input_size

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = 5 * torch.mul(x, self.params['Synapse_W']) - self.params['Synapse_q']
        x = torch.sigmoid(x)

        # Dendritic
        x = torch.prod(x, 3) #prod 

        # Membrane
        x = torch.sum(x, 2)

        # Soma
        x = 5 * (x - 0.5)

        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class DNM_multiple(nn.Module):   # MDNN * 2
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(DNM_multiple, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = DNM_Linear(input_size, hidden_size, M)
        self.DNM_Linear2 = DNM_Linear(hidden_size, out_size, M)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        out = self.DNM_Linear2(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = x.float()
        x = self.l1(x)
        x = torch.relu(self.l2(x))
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
