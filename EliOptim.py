import torch

class EliOptimizer():
    """Fast second-order optimizer.

    The optimizer uses Bayesian Linear regression to estimate the diagonal of
    the Hessian.  The result is a second-order optimizer, that updates with
    Saddle Free Newton Raphson, but also has low computational cost.

    Arguments:
        gamma: decay (default: 0.7)
        lamb_1: dampening (default: 0.3)
        lamb_2: regularizer (default: 1)
        lr: learning rate (default: 1e-4)
        tmin: burn-in period (default: 30)
    """


    def __init__(self, parameters, gamma = 0.9, lamb_1 = 0.1, lamb_2 = 1, lr = 1e-4, tmin = 30, running_mean = True, eps = 10**(-12):

        self.param_groups = []
        self.a_groups = []
        self.b_groups = []
        self.d_groups = []
        self.e_groups = []

        for param in parameters:
            self.param_groups.append(param)
            self.a_groups.append(torch.zeros(param.shape, dtype = torch.float64).to(param.device))
            self.b_groups.append(torch.zeros(param.shape, dtype = torch.float64).to(param.device))
            self.d_groups.append(torch.zeros(param.shape, dtype = torch.float64).to(param.device))
            self.e_groups.append(torch.zeros(param.shape, dtype = torch.float64).to(param.device))
        self.c = torch.zeros(1, dtype = torch.float64).to(param.device)

        self.lr = lr
        self.gamma = gamma
        self.lamb_1 = lamb_1
        self.lamb_2 = lamb_2
        self.t = 0
        self.eps = eps
        self.tmin = tmin
        self.running_mean = running_mean


    def step(self):
        """Performs a single optimization step."""

        self.t += 1
        self.c = self.gamma * self.c + 1

        for param, a, b, d, e in zip(self.param_groups, self.a_groups, self.b_groups, self.d_groups, self.e_groups):

            a.data = self.gamma * a.data + param.data**2
            b.data = self.gamma * b.data + param.data
            d.data = self.gamma * d.data + param.data * param.grad.data
            e.data = self.gamma * e.data + param.grad.data

            if self.t < self.tmin:
                 param.data -= self.lr * param.grad.data
            else:
                h = torch.abs((self.c * d - b * e) / (a * self.c - b**2 + self.eps))
                if self.running_mean:
                    param.data -= self.lamb_1 * (e / self.c) / (h + self.lamb_2)
                else:
                    param.data -= self.lamb_1 * param.grad.data / (h + self.lamb_2)

    def zero_grad(self):
        """Clears the gradients of all optimized"""
        for param in self.param_groups:
            param.grad.detach_()
            param.grad.zero_()
