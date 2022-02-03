from Euler.param import Param
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.set_printoptions(precision=3, linewidth=100000)
pp = print

"""
Partie visible
"""


def init_sod(param: Param):
    x = param.xs
    rho = tf.where(x < 0.5, 1., 0.125)[tf.newaxis, :]
    P = tf.where(x < 0.5, 1., 0.1)[tf.newaxis, :]
    rhoV = tf.zeros([1, param.nx])
    # Energy
    E = P / (param.gamma - 1)  # + 0.5 * rhoV * V
    return tf.stack([rho, rhoV, E], axis=2)


def init_random_sod(param: Param, batch_size: int, minimum_E=0.05, maximal_jump_E=2, minimum_rho=0.05, maximal_jump_rho=2):
    rho = generate_Riemann_Problem(param, batch_size, minimum_rho, maximal_jump_rho)
    E = generate_Riemann_Problem(param, batch_size, minimum_E, maximal_jump_E)
    rhoV = tf.zeros([batch_size, param.nx])
    return tf.stack([rho, rhoV, E], axis=2)


def init_periodic(param: Param,
                  batch_size,
                  continuousPart_scale=1,
                  discontinuousPart_scale=1,
                  minimum_P=0.5,
                  minimum_rho=0.5
                  ):

    genParam = GenParam(param, batch_size, True)
    genParam.continuousPart_scale = continuousPart_scale
    genParam.discontinuousPart_scale = discontinuousPart_scale
    genParam.minimum_P = minimum_P
    genParam.minimum_rho = minimum_rho

    generator = FuncGenerator(genParam, param)
    return generator.init_W()


def init_non_periodic(param: Param,
                  batch_size,
                  continuousPart_scale=1,
                  discontinuousPart_scale=1,
                  minimum_P=0.5,
                  minimum_rho=0.5
                      ):

    genParam = GenParam(param, batch_size, False)
    genParam.continuousPart_scale = continuousPart_scale
    genParam.discontinuousPart_scale = discontinuousPart_scale
    genParam.minimum_P = minimum_P
    genParam.minimum_rho = minimum_rho

    generator = FuncGenerator(genParam, param)
    return generator.init_W()


class GenParam:

    def __init__(self,
                 param: Param,
                 batch_size: int,
                 is_periodic: bool
                 ):
        self.param = param
        self.batch_size = batch_size  # au moins 25 pour certains tests (on affiche 5*5 images)

        self.is_periodic = is_periodic

        # loop et changing
        self.continuousPart_scale = 1.
        self.discontinuousPart_scale = 1.
        # si !=None, alors les 2 paramètres précédents seront ignoré.
        # à la place, la partie discontinue sera un ratio de la partie continue
        self.discountinousPart_ratio = None

        # partie continue de loop et changing
        self.fourierNbCoef = 2

        # uniquement les discontinuités de changing
        self.changing_discountMinSpacing = 0.3
        self.changing_discountMaxSpacing = 0.7
        assert self.changing_discountMinSpacing < self.changing_discountMaxSpacing

        self.minimum_P = 0.5
        self.minimum_rho = 0.5


class FuncGenerator:

    def __init__(self, genParam: GenParam, param: Param):
        self.genParam = genParam
        self.param = param

    def generate_fourier(self, batch_size=None):
        param = self.param

        if batch_size is None:
            batch_size = self.genParam.batch_size

        nbFourierCoef = self.genParam.fourierNbCoef
        # scale=self.genParam.fourierCoefRange
        # set up the Fourier modes

        n = tf.range(1., nbFourierCoef + 1)[tf.newaxis, :, tf.newaxis]
        # on divise par n pour que les hautes fréquences soient moins présentes
        an = tf.random.uniform(minval=-1, maxval=1, shape=(batch_size, nbFourierCoef, 1)) / n
        bn = tf.random.uniform(minval=-1, maxval=1, shape=(batch_size, nbFourierCoef, 1)) / n

        x = param.xs[tf.newaxis, tf.newaxis, :]
        nu = 2 * np.pi / (param.xmax - param.xmin)
        res = tf.reduce_sum(an * tf.cos(n * x * nu) + bn * tf.sin(n * x * nu), axis=1)
        return res

    # trop difficile de faire cela en tensorflow

    def generate_changing_discontinuity(self):
        param = self.param

        min_space_between_discont = int(param.nx * self.genParam.changing_discountMinSpacing)
        res = np.zeros([self.genParam.batch_size, param.nx])
        max_space_between_discont = int(param.nx * self.genParam.changing_discountMaxSpacing)

        for i in range(self.genParam.batch_size):
            lieu = 0
            # les lieux de sauts sont entre [min_space_between_discont,nx-min_space_between_discont]
            while lieu < param.nx - min_space_between_discont:
                next_lieu = lieu + np.random.randint(min_space_between_discont, max_space_between_discont)
                res[i, lieu:next_lieu] = np.random.uniform(-1, 1)
                lieu = next_lieu
        return res

    def generate_two_changing_discontinuities_with_same_location(self):
        param = self.param

        min_space_between_discont = int(param.nx * self.genParam.changing_discountMinSpacing)
        res0 = np.zeros([self.genParam.batch_size, param.nx])
        res1 = np.zeros([self.genParam.batch_size, param.nx])
        max_space_between_discont = int(param.nx * self.genParam.changing_discountMaxSpacing)

        for i in range(self.genParam.batch_size):
            lieu = 0
            # les lieux de sauts sont entre [min_space_between_discont,nx-min_space_between_discont]
            while lieu < param.nx - min_space_between_discont:
                next_lieu = lieu + np.random.randint(min_space_between_discont, max_space_between_discont)
                jump = np.random.uniform(-1, 1)
                res0[i, lieu:next_lieu] = jump * np.random.uniform(0.1, 1.5)
                res1[i, lieu:next_lieu] = jump * np.random.uniform(0.1, 1.5)
                lieu = next_lieu
        return res0, res1

    def generate_func_continuous(self, batch_size):

        continuity = self.generate_fourier(batch_size=batch_size)
        continuity_size = tf.random.uniform(minval=-self.genParam.continuousPart_scale, maxval=self.genParam.continuousPart_scale, shape=[batch_size])
        continuity = continuity * continuity_size[:, tf.newaxis]

        return continuity

    def init_W(self):
        param = self.param
        batch_size = self.genParam.batch_size

        """Les parties  continues de rho et P ont le même ordre de grandeur"""
        continuity_sizes = tf.random.uniform(minval=0, maxval=self.genParam.continuousPart_scale, shape=[batch_size, 1])
        continuous_rho = self.generate_func_continuous(batch_size) * continuity_sizes * tf.random.uniform(minval=0.8, maxval=1.2, shape=[batch_size, 1])
        continuous_P = self.generate_func_continuous(batch_size) * continuity_sizes * tf.random.uniform(minval=0.8, maxval=1.2, shape=[batch_size, 1])

        """ Les parties discontinues de rho et P on le même ordre de grandeur,
        et les sauts sont au même endroits, et dans le même sens"""
        discontinuous_rho, discontinuous_P = self.generate_two_changing_discontinuities_with_same_location()

        if self.genParam.discountinousPart_ratio is not None:
            discont_sizes = continuity_sizes * self.genParam.discountinousPart_ratio * tf.random.uniform(minval=0.8, maxval=1.2, size=[self.genParam.batch_size, 1])
        else:
            discont_sizes = tf.random.uniform(minval=0, maxval=self.genParam.discontinuousPart_scale, shape=[batch_size, 1])

        discontinuous_rho = discontinuous_rho * discont_sizes * np.random.uniform(0.8, 1.2, size=[self.genParam.batch_size, 1])
        discontinuous_P = discontinuous_P * discont_sizes * np.random.uniform(0.8, 1.2, size=[self.genParam.batch_size, 1])

        rho = discontinuous_rho + continuous_rho
        P = discontinuous_P + continuous_P

        scale = self.genParam.continuousPart_scale + self.genParam.discontinuousPart_scale

        mini = tf.reduce_min(rho, axis=1) - tf.random.uniform(minval=self.genParam.minimum_rho, maxval=1, shape=[self.genParam.batch_size]) * scale
        rho -= mini[:, tf.newaxis]

        miniP = tf.reduce_min(P, axis=1) - tf.random.uniform(minval=self.genParam.minimum_P, maxval=1, shape=[self.genParam.batch_size]) * scale
        P -= miniP[:, tf.newaxis]

        rhoV = tf.zeros([self.genParam.batch_size, param.nx])
        """
        Attention, la formule de l'énergie c'est:
                    P / (param.gamma - 1) + 0.5 * rhoV * V
        Mais on décide de toujours partir avec des vitesses nulles
        """
        E = P / (param.gamma - 1)  # + 0.5 * rhoV * V

        if self.genParam.is_periodic:
            rho = self.periodize(rho)

        return tf.stack([rho, rhoV, E], axis=2)

    def periodize(self, res):
        amplitudes = (res[:, -1] - res[:, 0])[:, tf.newaxis]
        affine = self.param.xs[tf.newaxis, :] / (self.param.xmax - self.param.xmin) * amplitudes
        res -= affine
        return res


def generate_Riemann_Problem(param: Param, batch_size: int, minimum, maximal_jump):
    res = np.zeros([batch_size, param.nx])
    for i in range(batch_size):
        w_left = np.random.uniform(minimum, 1)
        w_right = np.random.uniform(w_left * 1.1, w_left * 1.1 + maximal_jump)
        res[i, :param.nx // 2] = w_left
        res[i, param.nx // 2:] = w_right
    return res


""" TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST   """


def test_components():
    param = Param()
    genParam = GenParam(param, 64, False)
    generator = FuncGenerator(genParam, param)

    discount_changing0, discount_changing1 = generator.generate_two_changing_discontinuities_with_same_location()
    countinuity = generator.generate_fourier()

    fig, axs = plt.subplots(5, 2, sharey="all")
    for i in range(5):
        axs[i, 0].plot(discount_changing0[i, :])
        axs[i, 0].plot(discount_changing1[i, :])
        axs[i, 1].plot(countinuity[i, :])

    axs[0, 0].set_title("2 discount")
    axs[0, 1].set_title("continuity")

    plt.show()


def test_W_init():
    param = Param()
    W_periodic = init_periodic(param, 20)
    W_non_periodic = init_non_periodic(param, 20)
    W_sod = init_sod(param)

    fig, axs = plt.subplots(5, 3, sharey="all")
    for i in range(5):
        axs[i, 0].plot(W_periodic[i, :, 0], label="rho periodic")
        axs[i, 1].plot(W_non_periodic[i, :, 0], label="rho non periodic")
        axs[i, 2].plot(W_sod[0, :, 0], label="rho sod")

    for i in range(5):
        axs[i, 0].plot(W_periodic[i, :, 2], label="E periodic")
        axs[i, 1].plot(W_non_periodic[i, :, 2], label="E non periodic")
        axs[i, 2].plot(W_sod[0, :, 2], label="E sod")

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 2].legend()

    plt.show()


if __name__ == "__main__":
    # test_components()
    test_W_init()
