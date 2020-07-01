import torch
import torch.nn as nn


class StochasticRosenbrock(nn.Module):

    def __init__(self, seed = None, noise = 1):
        if seed:
            torch.manual_seed(seed)
        self.noise = noise


    def forward(self,tensor):
        x, y = tensor
        self.epsilon = torch.rand(1) * self.noise
        return (1 - x) ** 2 + 100 * self.epsilon * (y - x ** 2) ** 2

    def x_start(self):
        x = torch.zeros((2,1), dtype = torch.float64)
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([1.,1.], dtype = torch.float64)
        fmin = torch.tensor([0.], dtype = torch.float64)
        return xmin, fmin
    
    def hessian(self, tensor):
        x1, x2 = tensor
        h1 = 2 + self.epsilon* 1200*x1**2 -  self.epsilon* 400*x2
        h2 = 200 * self.epsilon
        h12 = -400 * x1 * self.epsilon
        return torch.tensor([[h1, h12], [h12, h2]])

class Rosenbrock(nn.Module):

    # def __init__(self, seed = None, noise = 1):

    def forward(self,tensor):
        x, y = tensor
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def hessian(self, tensor):
        x1, x2 = tensor
        h1 = 2 + 1200*x1**2 - 400*x2
        h2 = 200
        h12 = -400 * x1
        return torch.tensor([[h1, h12], [h12, h2]])

    def x_start(self):
        x = torch.zeros((2,1), dtype = torch.float64)
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([1.,1.], dtype = torch.float64)
        fmin = torch.tensor([0.], dtype = torch.float64)
        return xmin, fmin


class Powell(nn.Module):

    # def __init__(self):


    def forward(self,tensor):
        x1, x2, x3, x4 = tensor
        return (x1 + 10*x2)**2 + 5*(x3-x4)**4 + (x2 - 2*x3)**4 + 10*(x1-x4)**4

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def hessian(self, tensor):
        x1, x2, x3, x4 = tensor
        H = torch.zeros(4,4, dtype=tensor.dtype)
        H[0,0] = 120 * (x1 - x4)**2 + 2
        H[0,1] = 20
        H[0,2] = 0
        H[0,3] = -120 * (x1 - x4)**2
        H[1,0] = H[0,1]
        H[1,1] = 12 * (x2 - 2*x3)**2 + 200
        H[1,2] = -24*(x2-2*x3)**2
        H[1,3] = 0
        H[2,0] = H[0,2]
        H[2,1] = H[1,2]
        H[2,2] = 48*(x2 - 2*x3)**2 + 60*(x3-x4)**2
        H[2,3] = -60*(x3-x4)**2
        H[3,0] = H[0,3]
        H[3,1] = H[1,3]
        H[3,2] = H[2,3]
        H[3,3] = 120*(x1-x4)**2 + 60*(x3-x4)**2
        return H


    def x_start(self):
        x = torch.tensor([3,-1,0,1], dtype = torch.float64)
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([0,0,0,0], dtype = torch.float64)
        fmin = torch.tensor([0.], dtype = torch.float64)
        return xmin, fmin
