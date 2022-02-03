from Euler.param import Param
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.set_printoptions(precision=3, linewidth=100000)
pp = print
#
# def init_periodic(param:Param,batch_size):
#     generator = FuncGenerator(param,batch_size,True)
#     X = generator()
#     return X[:,:,tf.newaxis]
#
#
# def init_non_periodic(param:Param,batch_size):
#     generator = FuncGenerator(param,batch_size,False)
#     X = generator()
#     return X[:,:,tf.newaxis]


class FuncGenerator:

    def __init__(self, param: Param, batch_size, periodic):
        self.param = param
        self.batch_size = batch_size  # au moins 25 pour certains tests (on affiche 5*5 images)

        self.periodic = periodic

        self.continuousPart_scale = 0.5

        # la partie discontinue sera un ratio de la partie continue
        # self.discountinousPart_ratio = 2.
        self.discountinousPart_ratio = 5.

        # partie continue
        self.fourierNbCoef = 3

        # espacement entre discontinuités
        self.changing_discountMinSpacing = 0.3
        self.changing_discountMaxSpacing = 0.7
        assert self.changing_discountMinSpacing < self.changing_discountMaxSpacing

    def generate_fourier(self):
        param = self.param

        n = tf.range(1, self.fourierNbCoef + 1, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
        # on divise par n pour que les hautes fréquences soient moins présentes
        an = tf.random.uniform(minval=-1, maxval=1, shape=(self.batch_size, self.fourierNbCoef, 1)) / n
        bn = tf.random.uniform(minval=-1, maxval=1, shape=(self.batch_size, self.fourierNbCoef, 1)) / n

        x_ = tf.range(param.xmin, param.xmax, param.dx, dtype=tf.float32)
        x = x_[tf.newaxis, tf.newaxis, :]
        nu = 2 * np.pi / (param.xmax - param.xmin)
        return tf.reduce_sum(an * tf.cos(n * x * nu) + bn * tf.sin(n * x * nu), axis=1)

    def generate_func_continuous(self):
        continuity = self.generate_fourier()
        continuity_size = tf.random.uniform(minval=-self.continuousPart_scale, maxval=+self.continuousPart_scale, shape=[self.batch_size])
        return continuity * continuity_size[:, tf.newaxis]

    # donne des discontinuités très générique
    # trop difficile de faire cela en tensorflow

    def generate_discontinuity(self):
        param = self.param

        min_space_between_discont = int(param.nx * self.changing_discountMinSpacing)
        res = np.zeros([self.batch_size, param.nx])
        max_space_between_discont = int(param.nx * self.changing_discountMaxSpacing)

        for i in range(self.batch_size):
            lieu = 0
            # les lieux de sauts sont entre [min_space_between_discont,nx-min_space_between_discont]
            while lieu < param.nx - min_space_between_discont:
                next_lieu = lieu + np.random.randint(min_space_between_discont, max_space_between_discont)
                res[i, lieu:next_lieu] = np.random.uniform(-1, 1)
                lieu = next_lieu

        if self.periodic:
            amplitudes = (res[:, -1] - res[:, 0])[:, tf.newaxis]
            affine = self.param.xs[tf.newaxis, :] / (self.param.xmax - self.param.xmin) * amplitudes
            res -= affine

        return res

    def __call__(self):
        continuity_sizes = tf.random.uniform(minval=0, maxval=self.continuousPart_scale, shape=[self.batch_size, 1])
        discont_sizes = continuity_sizes * self.discountinousPart_ratio * tf.random.uniform(minval=0.8, maxval=1.2, shape=[self.batch_size, 1])
        res = self.generate_func_continuous() * continuity_sizes + self.generate_discontinuity() * discont_sizes
        return res[:, :, tf.newaxis]


""" TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST   """


def test():
    param = Param()
    generator = FuncGenerator(param, 64, False)
    X = generator()
    print(X.shape)

    fig, axs = plt.subplots(5, 5, sharex="all", sharey="all")
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.plot(param.xs, X[i, :])

    plt.show()


if __name__ == "__main__":
    test()
