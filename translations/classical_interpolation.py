import scipy
from scipy.interpolate import griddata
import numpy as np
import tensorflow as tf
from translations.initial_conditions import Param, FuncGenerator
import matplotlib.pyplot as plt


def test():
    param = Param(100)

    def one_method(y, method, ax):

        y_init=y+0.
        x_init=param.xs+0.

        x = param.xs+0.
        ax.set_title(method)

        def shift_func(x, y, epsilon):
            x_shift = x + epsilon
            y_shift = griddata(param.xs, y, x_shift, method=method,fill_value=y[0])
            return x_shift, y_shift

        nb_steps = 10
        for i in range(nb_steps):
            if i % 2 == 0:
                epsilon = +np.random.uniform(0, 7e-3)
            else:
                epsilon = -np.random.uniform(0, 7e-3)

            x, y = shift_func(x, y, epsilon)
            ax.plot(x, y, "r")


        ax.plot(x_init, y_init, "k")


    generator = FuncGenerator(param)
    Y = generator()
    y = Y[0, :].numpy()

    fig,axs=plt.subplots(3,1)
    one_method(y,"nearest",axs[0])
    one_method(y,"linear",axs[1])
    one_method(y,"cubic",axs[2])
    plt.tight_layout()
    plt.show()

def example_1D():
    x=tf.linspace(0,1,100)
    y=tf.sin(x).numpy()
    x_shift=np.linspace(0.01,0.101,100)
    y_shift=griddata(x,y,x_shift)
    #
    # plt.plot(x,y)
    # plt.plot(x_shift,y_shift,".-")
    # plt.show()

def example_doc():
    def func(x, y):
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

    points=np.random.uniform(-1,1,[1000,2])
    values = func(points[:, 0], points[:, 1])
    grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)

    plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
    plt.show()

#example_doc()
test()
#example_1D()