import torch
import numpy as np


class Rosenbrock():

    def __init__(self, seed = None):
        if seed:
            torch.manual_seed(seed)
        self.xstart = torch.randn((2,1)) * 5

    def val(self,tensor):
        x, y = tensor
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([1,1])
        fmin = torch.tensor(0)
        return xmin, fmin

class Powell():

    def __init__(self, seed = None):
        if seed:
            torch.manual_seed(seed)
        self.xstart = torch.randn((4,1)) * 5

    def val(self,tensor):
        x1, x2, x3, x4 = tensor
        return (x1 + 10*x2)**2 + 5*(x3-x4)**4 + (x2 - 2*x3)**4 + 10*(x1-x4)**4

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([0,0,0,0])
        fmin = torch.tensor(0)
        return xmin, fmin
    
class StochasticRosenbrock():

    def __init__(self, seed = None, noise = 1):
        if seed:
            torch.manual_seed(seed)
        self.xstart = torch.randn((2,1)) * 0
        self.noise = noise

    def val(self,tensor):
        x, y = tensor
        epsilon = torch.rand(1) * self.noise
        return (1 - x) ** 2 + 100 * epsilon * (y - x ** 2) ** 2

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([1,1])
        fmin = torch.tensor(0)
        return xmin, fmin

class StochasticPowell():

    def __init__(self, seed = None, noise = 1):
        if seed:
            torch.manual_seed(seed)
        self.xstart = torch.randn((4,1)) * 5
        self.noise = noise

    def val(self,tensor):
        x1, x2, x3, x4 = tensor
        epsilon1 = torch.rand(1) * self.noise
        epsilon2 = torch.rand(1) * self.noise
        epsilon3 = torch.rand(1) * self.noise
        epsilon4 = torch.rand(1) * self.noise

        return (x1 + 10*epsilon1*x2)**2 + 5*epsilon2*(x3-x4)**4 + (x2 - 2*epsilon3*x3)**4 + 10*epsilon4*(x1-x4)**4

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([0,0,0,0])
        fmin = torch.tensor(0)
        return xmin, fmin

class ThreeHumpCamel():

    def __init__(self, seed = None):
        self.xstart = torch.randn((2,1))

    def val(self,tensor):
        x, y = tensor
        return 2*x**2 - 1.05*x**4 + (1/6)*x**6  + x*y + x**2

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x
    
    def min(self):
        xmin = torch.tensor([0,0])
        fmin = torch.tensor(0)
        return xmin, fmin

class SixHumpCamel():

    def __init__(self, seed = None):
        self.xstart = torch.randn((2,1))

    def val(self,tensor):
        x, y = tensor
        return (4-2.1*x**2 + (1/3)*x**4) * x**2 + x*y + (-4 + 4*x**2) * x**2

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

class Booth():

    def __init__(self, seed = None):
        if seed:
            torch.manual_seed(seed)

        self.xstart = torch.randn((2,1)) * 10

    def val(self,tensor):
        x, y = tensor
        return (x+2*y-7)**2 + (2*x + y -5)**2

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([1,3])
        fmin = torch.tensor(0)
        return xmin, fmin

class Branin():

    def __init__(self, seed = None):
        if seed:
            torch.manual_seed(seed)

        self.xstart = torch.randn((2,1)) * 2

    def val(self,tensor):
        x, y = tensor
        a = 1
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1/(8*np.pi)

        return a * (y - b*x**2 + c*x - r)**2 + s*(1-t)*torch.cos(x) + s

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def min(self):
        xmin = "None"
        fmin = 0.397887
        return xmin, fmin

class Coville():

    def __init__(self, seed = None):
        if seed:
            torch.manual_seed(seed)

        self.xstart = torch.randn((4,1)) * 10

    def val(self,tensor):
        x1,x2,x3,x4 = tensor

        return 100*(x1**2-x2)**2 + (x1-1)**2 + (x3-1)**2 + 90*(x3**2-x4)**2 + 10.1*((x2-1)**2+(x4-1)**2) + 19.8*(x2-1)*(x4-1)

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([1,1,1,1])
        fmin = 0
        return xmin, fmin

class Matyas():

    def __init__(self, seed = None):
        if seed:
            torch.manual_seed(seed)

        self.xstart = torch.randn((2,1)) * 10

    def val(self,tensor):
        x1,x2 = tensor

        return 0.2*(x1**2 + x2**2) - 0.48*x1*x2

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def min(self):
        xmin = torch.tensor([0,0])
        fmin = 0
        return xmin, fmin


class quadratic():
    def __init__(self, size = 2, seed = None):

        self.size = size

        if seed:
            torch.manual_seed(seed)

        self.A = torch.randn((size,size))
        self.A = self.A @ self.A.T
        self.b = torch.randn((size,1))

        # choosing starting locations

        self.xstart = 20 * torch.randn((self.size,1))

    def set_A_b(self, A, b):
        self.size = A.shape[0]
        self.A = A,
        self.b = b

    def x_start(self):

        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def val(self, x):
        # x = x.T
        return 0.5*x.T@self.A@x - self.b.T@x

    def min(self):
        x_min = self.A.inverse()@self.b
        f_min = self.val(x_min)
        return x_min, f_min



class Quadratic_data():
    def __init__(self, size = 2, n_data = 200, seed = None):

        self.size = size
        self.n_data = n_data

        self.seed = seed

        if seed != None:
            torch.manual_seed(seed)

        self.x_min = torch.randn((size, n_data)) * 5
        self.A = torch.randn((size,size)) * 10
        self.A = self.A @ self.A.T
        self.b = self.A @ self.x_min

        # choosing starting locations
        self.xstart = 20 * torch.randn((self.size,1))

    def set_A_b(self, A, b):
        self.size = A.shape[0]
        self.A = A,
        self.b = b

    def x_start(self):
        x = self.xstart.detach()
        x.requires_grad = True
        return x

    def val(self, x, batch_size = 30):

        index = np.random.randint(self.n_data, size=batch_size)
        return 0.5*x.T@self.A@x - self.b[:,index].T@x

    # def minimum(self):
    #     x_min = self.x_min
    #     f_min = self.val(x_min)
    #     return x_min, f_min
