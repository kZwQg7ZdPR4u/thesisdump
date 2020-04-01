import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from IPython.display import HTML


def loss_plot(log, plot_type = "linear"):

    optimizer_names = log["optimizer_names"]

    plt.figure(figsize=(20,10))
    for optimizer_name in optimizer_names:
        t = log[optimizer_name]["t"]
        x = log[optimizer_name]["x"]
        y = log[optimizer_name]["y"]
        plt.xlabel("iterations")
        plt.ylabel("f(x),    lower=better")
        plt.yscale(plot_type)
        plt.xscale(plot_type)
        plt.plot(t,y, label= optimizer_name)
    plt.legend()
    plt.show()

def plot_basis(log):

    x = log["x"]
    y = log["y"]
    problem = log["problem"]
    n_iterations = log["n_iterations"]
    xmin = torch.tensor(log["xmin"])
    ymin = problem.val(xmin)

    t = log["t"]

    width = max(x) - min(x)
    x_func = torch.tensor(np.linspace(min(x) - width, max(x) + width, 50))
    y_func = problem.val(x_func)


    fig = plt.figure(figsize=(10,6))
    ax_main = fig.add_subplot(1, 1, 1)

    # Initializing lines and locations
    ax_main.plot(x_func, y_func)
    SGD_path, = ax_main.plot([], [],"o", lw=2, label="SGD")
    estimated_xmin, = ax_main.plot([], [], "o", lw=2, label = "estimated minimum")
    ax_main.legend()

    def init():
    # Background and legends
        SGD_path.set_data([], [])
        estimated_xmin.set_data([], [])
        return [SGD_path, estimated_xmin]

    def animate(i):
        SGD_path.set_data(x[i], y[i])
        estimated_xmin.set_data(xmin[i], ymin[i])
        return [SGD_path, estimated_xmin]

    anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=n_iterations, interval=100, blit=True, repeat=True)
    plt.show()

    # HTML(anim.to_html5_video())

    return anim


def plot_gradfit(log):
    x = log["x"]
    g = log["g"]


    print(g)
    recent = log["recent"]
    problem = log["problem"]
    n_iterations = log["n_iterations"]

    fig = plt.figure(figsize=(10,6))
    ax_main = fig.add_subplot(1, 1, 1)
    ax_main.set_xlim(min(x), max(x))
    ax_main.set_ylim(min(g), max(g))


    # Initializing lines and locations
    g_points, = ax_main.plot([], [],"o", lw=2, label="gradients")
    # ax_main.legend()

    def init():
    # Background and legends
        g_points.set_data([], [])
        return [g_points]

    def animate(i):
        g_points.set_data(x[i:recent+i], g[i:recent+i])
        return [g_points]

    anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=n_iterations, interval=100, blit=True, repeat=True)

    plt.show()

def ellipse_plotting(log):

    betas = log["beta"]
    Vs = log["V"]
    n_iterations = log["n_iterations"]

    width = []
    height = []
    theta = []

    for V in Vs:
        # print("a V:", V)
        w, h, t =  cov_ellipse(V.numpy(), nsig=1)
        width.append(w[0])
        height.append(h[0])
        theta.append(t)

    xy = []
    for beta in betas:
        xy.append((beta[0].item(), beta[1].item()))

    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-400, 400)
    ax.autoscale()

    e1 = Ellipse(xy=(0, 0), width=10, height=10, angle=60)
    # e2 = Ellipse(xy=(0.8, 0.8), width=0.5, height=0.2, angle=100)
    ax.add_patch(e1)
    # ax.add_patch(e2)
    #
    def init():
        e1.set_visible(False)
        return [e1]
    #
    def animate(i):
        if i == 1:
            e1.set_visible(True)
        # print(xy[i])
        # e1.xy[0] = (1,1)
        e1.width = width[i]
        e1.height = height[i]
        e1.angle = theta[i]
        #

        print("xy, width, height, theta", xy[i], width[i], height[i], theta[i])
        # e1.height = 0.2
        # e1.angle = 60
        # e1.xy = xy[i]
        # e1.width = width[i]
        # e1.height = height[i]
        # e1.angle = theta[i]
        return [e1]
    #
    anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=n_iterations, interval=100, blit=True, repeat=True)
    plt.show()


from scipy.stats import norm, chi2

def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation
