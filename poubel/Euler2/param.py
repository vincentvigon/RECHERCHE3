import tensorflow as tf
import matplotlib.pyplot as plt
pp=print

class Param:

    BC_periodic = "periodic"
    BC_neumann = "neumann"
    BC_reflexive = "reflexive"

    def __init__(self,
                 nx=2000,  # number of points in grid
                 nx_ratio=10,
                 problem="burger", # "burger", "euler"
                 BC="neumann",
                 xmin=0.,
                 xmax=1.,
                 CFL=0.5,
                 gamma=1.4, #uniquement pour Euler
                 dt_over_dx=0.3,
                 ):

        self.nx = nx
        self.nx_ratio = nx_ratio
        self.problem=problem
        self.BC = BC

        assert xmin < xmax
        self.xmin = xmin
        self.xmax = xmax

        self.CFL = CFL

        assert gamma > 1, "gamma doit être >1"
        self.gamma = gamma


        self.xs = tf.linspace(self.xmin, self.xmax, self.nx)

        """ dt/dx  doit être suffisamment petit pour que la CFL soit toujours satisfaite. Il y a un assert qui le vérifie dans la fonction Flux_HLL"""
        self.dt_over_dx = dt_over_dx
        #self.Nmax = Nmax

        self.dx = (self.xmax - self.xmin) / self.nx
        self.dt = self.dt_over_dx * self.dx

        """COARSE GRID
        nx_coarse c'est à peu pret nx/nx_ratio, mais le padding=valid de la convolution glissante rabote un peu.
        """
        self.nx_coarse = Projecter(self.nx_ratio).projection_1D(tf.linspace(0., 1, self.nx)).shape[0]
        self.dx_coarse = (self.xmax - self.xmin) / self.nx_coarse
        self.dt_over_dx_coarse = self.dt / self.dx_coarse

        print(f"Param initialised with, nx={self.nx}, nx_coarse={self.nx_coarse}")


        if problem=="burger":
            self.F_fn=lambda a:a**2/2
        else:
            raise Exception("euler: TODO")

        self.D_fn_rusanov=lambda a,b: tf.maximum(tf.abs(a), tf.abs(b))


class Projecter:

    def __init__(self, nx_ratio: int):
        self.nx_ratio = nx_ratio
        width = 2 * nx_ratio
        sigma = 2.
        # mieux si width impair
        m = (width - 1) // 2
        y = tf.range(-m, m + 1, 1.)
        self.mask = tf.exp(-(y * y) / (2. * sigma * sigma))
        """normalization"""
        self.mask /= tf.reduce_sum(self.mask)

    def projection_1D(self, tensor):
        #très peu utilisé, pas besoin d'accélerer
        assert len(tensor.shape) == 1, "input must be a 1D tensor"
        stride = self.nx_ratio
        conv = tf.nn.conv1d(tensor[tf.newaxis, :, tf.newaxis], self.mask[:, tf.newaxis, tf.newaxis], stride=stride, padding="VALID")
        return conv[0, :, 0]


    @tf.function
    def projection_3D(self, tensor):
        print("traçage de la projection 3d pour la shape",tensor.shape)

        assert len(tensor.shape) == 3, "input must be a 3D tensor"
        batch_size, nx, nb_channels = tensor.shape
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

    @tf.function
    def projection_4D(self, tensor):
        print("traçage de la projection 4d pour la shape",tensor.shape)
        assert len(tensor.shape) == 4, "input must be a 4D tensor"
        nb_t,batch_size, nx, nb_channels = tensor.shape
        stride = self.nx_ratio
        """
        On veut effectuer une convolution sur la dimension centrale (space_x) sans mélanger les channels.
        Pour éviter le mélange des channel on fait une transposition qui transforme les channels en élément d'un batch
        """
        tensor = tf.transpose(tensor, [0, 1, 3, 2])

        tensor = tf.reshape(tensor, [nb_t*batch_size * nb_channels, nx, 1])
        conv = tf.nn.conv1d(tensor, self.mask[:, tf.newaxis, tf.newaxis], stride=stride, padding="VALID")
        conv = tf.reshape(conv, [nb_t,batch_size, nb_channels, -1])
        conv = tf.transpose(conv, [0,1, 3, 2])
        return conv


def test_gaussian_smoothing(dim_4):
    batch_size=1
    nx=100

    batch_function =  tf.concat([tf.zeros([batch_size,nx//2,3]),tf.ones([batch_size,nx//2,3])],axis=1)# (batch_size,nx,3)
    if dim_4:
        batch_function=batch_function[tf.newaxis,:,:,:]

    ratio=3
    projecter=Projecter(ratio)

    if not dim_4:
        proj=projecter.projection_3D(batch_function)
    else:
        proj = projecter.projection_4D(batch_function)
        proj=proj[0,:,:,:]
        batch_function=batch_function[0,:,:,:]

    fig,(ax1,ax2)=plt.subplots(2,1)

    for i in [0,1,2]:
        ax1.plot(batch_function[0,:,i])
        ax1.set_title("full resolution")

        ax2.plot(proj[0,:,i])
        ax2.set_title(f"resolution/{ratio}")

    plt.show()


if __name__=="__main__":
    test_gaussian_smoothing(True)
