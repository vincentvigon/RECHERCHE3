import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Param:

    BC_periodic = "periodic"
    BC_neumann = "neumann"
    BC_reflexive = "reflexive"

    def __init__(self,
                 nx=2000, # number of points in grid
                 nx_ratio=10,
                 BC_solver="periodic",
                 BC_model=None,  # None=> BC_model=BC_solver
                 xmin=0.,
                 xmax=1.,
                 CFL=0.5,
                 gamma=1.4,
                 dt_over_dx=0.1,
                 order=1,
                 HLLC=True
                 ):

        self.nx = nx
        self.nx_ratio = nx_ratio

        self.BC_solver = BC_solver
        if BC_model is None:
            self.BC_model = BC_solver
        else:
            self.BC_model = BC_model

        self.xmin = xmin
        self.xmax = xmax
        assert xmin < xmax
        self.CFL = CFL

        assert gamma > 1, "gamma doit être >1"
        self.gamma = gamma
        """ dt/dx  doit être suffisamment petit pour que la CFL soit toujours satisfaite. Il y a un assert qui le vérifie dans la fonction Flux_HLL"""
        self.dt_over_dx = dt_over_dx
        #self.Nmax = Nmax

        self.dx = (self.xmax - self.xmin) / self.nx
        self.dt = self.dt_over_dx * self.dx

        """COARSE GRID
        nx_coarse c'est à peu pret nx/nx_ratio, mais le padding=valid de la convolution glissante rabote un peu.
        """
        self.nx_coarse = Projecter(self.nx_ratio, 64).projection_1D(np.linspace(0, 1, self.nx)).shape[0]
        self.dx_coarse = (self.xmax - self.xmin) / self.nx_coarse
        self.dt_over_dx_coarse = self.dt / self.dx_coarse

        # pour vérifier qu'on fait une augmentation de donnée de la bonne dimension.
        # il faut changer ce paramètre quand on modifie Var.get_augmentation
        self.augmentation_dim = 8

        self.order=order
        self.HLLC=HLLC


        print(f"Param initialised with, nx={self.nx}, nx_coarse={self.nx_coarse}")


#Ne fonctionne pas bien: il faudrait préciser la taille du tenseur apparamment
# @tf.function
# def projection_accelerated(tensor,param:Param,precision=32):
#     width = 2 * param.nx_ratio
#     sigma = 2.
#     k  = K("tf", precision)
#
#     # mieux si width impair
#     m = (width - 1) // 2
#     y = k.arange_float(-m, m + 1, 1)  # np.ogrid[-m:m + 1]
#     mask = k.exp(-(y * y) / (2. * sigma * sigma))
#     """normalization"""
#     mask /= k.sum(mask)
#     batch_size, nx, nb_channel = tensor.shape
#     nb_channels = tensor.shape[2]
#     stride = param.nx_ratio
#     """
#     On veut effectuer une convolution sur la dimension centrale (space_x) sans mélanger les channels.
#     Pour éviter le mélange des channel on fait une transposition qui transforme les channels en élément d'un batch
#     """
#     tensor = tf.transpose(tensor, [0, 2, 1])
#     tensor = tf.reshape(tensor, [batch_size * nb_channels, nx, 1])
#     conv = tf.nn.conv1d(tensor, mask[:, tf.newaxis, tf.newaxis], stride=stride, padding="VALID")
#     conv = tf.reshape(conv, [batch_size, nb_channels, -1])
#     conv = tf.transpose(conv, [0, 2, 1])
#     return conv



# ici j'ai fait que la procédure tf
class Projecter:

    def __init__(self, nx_ratio: int, precision: int):
        self.nx_ratio = nx_ratio
        k = self.k = K("tf", precision)
        width = 2 * nx_ratio
        sigma = 2.
        # mieux si width impair
        m = (width - 1) // 2
        y = k.arange_float(-m, m + 1, 1)  # np.ogrid[-m:m + 1]
        self.mask = k.exp(-(y * y) / (2. * sigma * sigma))
        """normalization"""
        self.mask /= k.sum(self.mask)

    def projection_1D(self, tensor):
        assert len(tensor.shape) == 1, "input must be a 1D tensor"
        tensor = self.k.convert(tensor)
        stride = self.nx_ratio
        conv = tf.nn.conv1d(tensor[tf.newaxis, :, tf.newaxis], self.mask[:, tf.newaxis, tf.newaxis], stride=stride, padding="VALID")
        return conv[0, :, 0]

    #Ne pas mettre @tf.function ici
    def projection(self, tensor):
        assert len(tensor.shape) == 3, "input must be a 3D tensor"
        batch_size, nx, nb_channel = tensor.shape
        nb_channels = tensor.shape[2]
        stride = self.nx_ratio
        """
        On veut effectuer une convolution sur la dimension centrale (space_x) sans mélanger les channels.
        Pour éviter le mélange des channel on fait une transposition qui transforme les channels en élément d'un batch
        """
        tensor = tf.transpose(tensor, [0, 2, 1])
        tensor = tf.reshape(tensor, [batch_size * nb_channels, nx, 1])
        conv = tf.nn.conv1d(tensor, self.mask[:, tf.newaxis, tf.newaxis], stride=stride, padding="VALID")
        conv = tf.reshape(conv, [batch_size, nb_channels, -1])
        conv = tf.transpose(conv, [0, 2, 1])
        return conv
