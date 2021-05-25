from Euler.param import Param
from Euler.backend import K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import pandas as pd
np.set_printoptions(precision=3, linewidth=100000)
pp = print

"""
Partie visible
"""

def init_sod(param:Param):
    k=K("tf",32)
    x = k.arange_float(param.xmin, param.xmax, param.dx)
    rho = k.where_float(x < 0.5, 1., 0.125)[k.newaxis, :]
    P = k.where_float(x < 0.5, 1., 0.1)[k.newaxis, :]
    rhoV = k.zeros_float([1, param.nx])
    # Energy
    E = P / (param.gamma - 1)  # + 0.5 * rhoV * V
    return k.stack([rho, rhoV, E], axis=2)


def init_random_sod(param:Param, batch_size:int, minimum_E=0.05, maximal_jump_E=2, minimum_rho=0.05, maximal_jump_rho=2):
    k=K("tf",32)
    rho = generate_Riemann_Problem(param,batch_size,k,minimum_rho,maximal_jump_rho)
    E = generate_Riemann_Problem(param, batch_size, k, minimum_E, maximal_jump_E)
    rhoV = k.zeros_float([batch_size, param.nx])
    return k.stack([rho, rhoV, E], axis=2)


def init_periodic(param:Param,
                  batch_size,
                  continuousPart_scale=1,
                  discontinuousPart_scale=1,
                  minimum_P=0.5,
                  minimum_rho=0.5
                  ):

    genParam=GenParam(param,"loop",batch_size)
    genParam.continuousPart_scale=continuousPart_scale
    genParam.discontinuousPart_scale=discontinuousPart_scale
    genParam.minimum_P=minimum_P
    genParam.minimum_rho=minimum_rho

    generator=FuncGenerator(genParam,param,K("tf",32))
    return generator.init_W()


def init_non_periodic(param: Param,
                  batch_size,
                  continuousPart_scale=1,
                  discontinuousPart_scale=1,
                  minimum_P=0.5,
                  minimum_rho=0.5
                  ):

    genParam = GenParam(param, "changing", batch_size)
    genParam.continuousPart_scale = continuousPart_scale
    genParam.discontinuousPart_scale = discontinuousPart_scale
    genParam.minimum_P = minimum_P
    genParam.minimum_rho = minimum_rho

    generator = FuncGenerator(genParam, param, K("tf", 32))
    return generator.init_W()





class GenParam:
    kind_loop = "loop"
    kind_changing = "changing"

    def __init__(self,
                 param: Param,
                 kind,
                 batch_size,
                 ):
        self.param = param
        self.kind = kind  # random_sod, sod, loop, changing
        self.batch_size = batch_size  # au moins 25 pour certains tests (on affiche 5*5 images)

        # loop et changing
        self.continuousPart_scale = 1
        self.discontinuousPart_scale = 1
        # si !=None, alors les 2 paramètres précédents seront ignoré.
        # à la place, la partie discontinue sera un ratio de la partie continue
        self.discountinousPart_ratio = None

        # partie continue de loop et changing
        self.fourierNbCoef = 2

        # uniquement les discontinuités de changing
        self.changing_discountMinSpacing = 0.3
        self.changing_discountMaxSpacing = 0.7
        assert self.changing_discountMinSpacing < self.changing_discountMaxSpacing

        # pour W_init; si initial_speed_factor=0, pas de vitesse initiale (comme sod)
        #self.initial_speed_factor = 0.

        self.minimum_P=0.5
        self.minimum_rho=0.5



class FuncGenerator:

    def __init__(self, genParam: GenParam, param: Param, k: K):
        self.genParam = genParam
        self.param = param
        self.k = k

    def generate_fourier(self, batch_size=None):
        k = self.k
        param = self.param

        if batch_size is None:
            batch_size = self.genParam.batch_size

        nbFourierCoef = self.genParam.fourierNbCoef
        # scale=self.genParam.fourierCoefRange
        # set up the Fourier modes

        n = k.arange_float(1, nbFourierCoef + 1)[k.newaxis, :, k.newaxis]
        # on divise par n pour que les hautes fréquences soient moins présentes
        an = k.random_uniform_float(-1, 1, shape=(batch_size, nbFourierCoef, 1)) / n
        bn = k.random_uniform_float(-1, 1, shape=(batch_size, nbFourierCoef, 1)) / n

        x_ = k.arange_float(param.xmin, param.xmax, param.dx)
        x = x_[k.newaxis, k.newaxis, :]
        nu = 2 * k.pi / (param.xmax - param.xmin)
        res = k.sum(an * k.cos(n * x * nu) + bn * k.sin(n * x * nu), axis=1)
        return res


    def generate_loop_dicontinuity(self):
        k = self.k
        """on définit le lieu des discontinuités"""
        discontinuity = self.generate_fourier()
        mini = k.min(discontinuity, axis=1)
        maxi = k.max(discontinuity, axis=1)
        # ici la procédure tf.random.uniform ne permet pas de faire cela
        coef = 0.8  # si on prend 1, les discontinuité pourront être plus rapprochées
        threshold = np.random.uniform(mini * coef, maxi * coef)[:, np.newaxis]
        threshold = k.convert(threshold)
        res = k.where_float(discontinuity > threshold, -1, 1)
        return res


    # un peu plus long que le mode loop, mais cela donne des discontinuités très générique
    # trop difficile de faire cela en tensorflow
    def generate_changing_discontinuity(self):
        k = self.k
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
        return k.convert(res)


    def generate_two_changing_discontinuities_with_same_location(self):
        k = self.k
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
                jump=np.random.uniform(-1, 1)
                res0[i, lieu:next_lieu] = jump*np.random.uniform(0.5, 1)
                res1[i, lieu:next_lieu] = jump*np.random.uniform(0.5, 1)
                lieu = next_lieu
        return k.convert(res0),k.convert(res1)

    # def generate_two_changing_discontinuities(self):
    #     k = self.k
    #     param = self.param
    #
    #     res = np.zeros([self.genParam.batch_size, param.nx])
    #
    #     for i in range(self.genParam.batch_size):
    #         w_left = 1 + 2 * np.random.random()
    #         w_mid = 1 + 2 * np.random.random()
    #         w_right = 1 + 2 * np.random.random()
    #         res[i, :param.nx // 4] = w_left
    #         res[i, param.nx // 4: 3 * param.nx // 4] = w_mid
    #         res[i, 3 * param.nx // 4:] = w_right
    #
    #     return k.convert(res)
    #
    # def generate_func(self):
    #     k = self.k
    #
    #     continuity = self.generate_fourier()
    #     continuity_size = k.random_uniform_float(-self.genParam.continuousPart_scale, self.genParam.continuousPart_scale, shape=[self.genParam.batch_size])
    #     continuity = continuity * continuity_size[:, k.newaxis]
    #
    #     if self.genParam.kind == "loop":
    #         discount = self.generate_loop_dicontinuity()
    #     elif self.genParam.kind == "changing":
    #         discount = self.generate_changing_discontinuity()
    #     else:
    #         raise Exception(f"kind: {self.genParam.kind} is not implemented. Possibility loop,changing")
    #
    #     if self.genParam.discountinousPart_ratio is not None:
    #         discount_size = continuity_size * k.random_uniform_float(-self.genParam.discountinousPart_ratio, +self.genParam.discountinousPart_ratio, shape=[self.genParam.batch_size])
    #     else:
    #         discount_size = k.random_uniform_float(-self.genParam.discontinuousPart_scale, self.genParam.discontinuousPart_scale, shape=[self.genParam.batch_size])
    #
    #     discount = discount * discount_size[:, k.newaxis]
    #
    #     return continuity + discount


    def generate_func_continuous(self, batch_size):
        k = self.k

        continuity = self.generate_fourier(batch_size=batch_size)
        continuity_size = k.random_uniform_float(-self.genParam.continuousPart_scale, self.genParam.continuousPart_scale, shape=[batch_size])
        continuity = continuity * continuity_size[:, k.newaxis]

        return continuity

    #
    # def generate_func_discontinuous(self, batch_size):
    #     k = self.k
    #
    #     continuity_size = k.random_uniform_float(-self.genParam.continuousPart_scale, self.genParam.continuousPart_scale, shape=[batch_size])
    #
    #     if self.genParam.kind == "loop":
    #         discount = self.generate_loop_dicontinuity()
    #     elif self.genParam.kind == "changing":
    #         discount = self.generate_changing_discontinuity()
    #     else:
    #         raise Exception(f"kind: {self.genParam.kind} is not implemented. Possibility loop,changing")
    #
    #     if self.genParam.discountinousPart_ratio is not None:
    #         discount_size = continuity_size * k.random_uniform_float(0.8, + 1.2, shape=[self.genParam.batch_size])
    #     else:
    #         discount_size = k.random_uniform_float(-self.genParam.discontinuousPart_scale, self.genParam.discontinuousPart_scale, shape=[self.genParam.batch_size])
    #
    #     discount = discount * discount_size[:, k.newaxis]
    #
    #     return discount



    def init_W(self):
        k = self.k
        param = self.param
        batch_size=self.genParam.batch_size

        """Les parties  continues de rho et P ont le même ordre de grandeur"""
        continuity_sizes = k.random_uniform_float(0, self.genParam.continuousPart_scale, shape=[batch_size,1])
        continuous_rho =  self.generate_func_continuous(batch_size)*continuity_sizes*k.random_uniform_float(0.8,1.2,shape=[batch_size,1])
        continuous_P = self.generate_func_continuous(batch_size)*continuity_sizes*k.random_uniform_float(0.8,1.2,shape=[batch_size,1])

        """ Les parties discontinues de rho et P on le même ordre de grandeur,
        et les sauts sont au même endroits, et dans le même sens"""
        if self.genParam.kind==GenParam.kind_changing:
            discontinuous_rho,discontinuous_P=self.generate_two_changing_discontinuities_with_same_location()
        else:
            discont=self.generate_loop_dicontinuity()*np.random.uniform(-1,1,size=[self.genParam.batch_size,1])
            discont0=discont*np.random.uniform(0.8,1.2,size=[self.genParam.batch_size,1])
            discont1=discont0*np.random.uniform(0.8,1.2,size=[self.genParam.batch_size,1])
            discontinuous_rho, discontinuous_P=discont0,discont1

        if self.genParam.discountinousPart_ratio is not None:
            discont_sizes = continuity_sizes*self.genParam.discountinousPart_ratio*np.random.uniform(0.8,1.2,size=[self.genParam.batch_size,1])
        else:
            discont_sizes = k.random_uniform_float(0, self.genParam.discontinuousPart_scale, shape=[batch_size,1])

        discontinuous_rho=discontinuous_rho*discont_sizes*np.random.uniform(0.8,1.2,size=[self.genParam.batch_size,1])
        discontinuous_P=discontinuous_P*discont_sizes*np.random.uniform(0.8,1.2,size=[self.genParam.batch_size,1])


        rho = discontinuous_rho + continuous_rho
        P = discontinuous_P + continuous_P

        scale = self.genParam.continuousPart_scale + self.genParam.discontinuousPart_scale

        mini = k.min(rho, axis=1) - k.random_uniform_float(self.genParam.minimum_rho, 1, shape=[self.genParam.batch_size]) * scale
        rho -= mini[:, k.newaxis]

        miniP = k.min(P, axis=1) - k.random_uniform_float(self.genParam.minimum_P, 1, shape=[self.genParam.batch_size]) * scale
        P -= miniP[:, k.newaxis]


        rhoV = k.zeros_float([self.genParam.batch_size, param.nx])
        """
        Attention, la formule de l'énergie c'est: 
                    P / (param.gamma - 1) + 0.5 * rhoV * V
        Mais on décide de toujours partir avec des vitesses nulles
        """
        E = P / (param.gamma - 1) #+ 0.5 * rhoV * V

        return k.stack([rho, rhoV, E], axis=2)
    #
    # def init_W_old(self):
    #     k = self.k
    #     param = self.param
    #
    #     """ computes the initial conditions for  rho,rhoV,E   """
    #
    #     x = k.arange_float(param.xmin, param.xmax, param.dx)
    #
    #     if self.genParam.kind == GenParam.kind_sod:  # Sod test case
    #         rho = self.genParam.sod_scale * k.where_float(x < 0.5, 1., 0.125)
    #         V = k.zeros_float([param.nx])
    #         P = self.genParam.sod_scale * k.where_float(x < 0.5, 1., 0.1)
    #         rhoV = rho * V
    #         E = P / (param.gamma - 1.0) + 0.5 * rhoV * V
    #         W = []
    #         for i in range(self.genParam.batch_size):
    #             W.append(k.stack([rho, rhoV, E], axis=1))
    #         return k.stack(W, axis=0)
    #
    #     else:
    #         # positif
    #         rho = self.generate_func()
    #         a0 = k.random_uniform_float(1e-2, 2, shape=[self.genParam.batch_size])[:, k.newaxis]
    #         mini = k.min(rho, axis=1)
    #         rho -= mini[:, k.newaxis]
    #         rho += a0
    #
    #         # todo
    #         Mach = 1  # 10**k.random_uniform_float(-1, 1, shape=[param.batch_size])  #avant -4,4
    #
    #         if self.genParam.initial_speed_factor > 0:
    #             V_mach = k.minimum(Mach, k.ones_float(self.genParam.batch_size)) * self.genParam.initial_speed_factor
    #             V = self.generate_func() * V_mach[:, k.newaxis]
    #         else:
    #             V = k.zeros_float([self.genParam.batch_size, param.nx])
    #
    #         # là je me suis permis de changer les formules qui me paraissaient bizarre
    #         T = self.generate_func()
    #         a0 = k.random_uniform_float(1e-2, 2, shape=[self.genParam.batch_size])[:, k.newaxis]
    #         mini = k.min(T, axis=1)
    #         T -= mini[:, k.newaxis]
    #         T += a0
    #         # pressure
    #         P_mach = k.minimum(1. / Mach, k.ones_float(self.genParam.batch_size))
    #         P = rho * T * P_mach[:, k.newaxis]
    #         # momentum
    #         rhoV = rho * V
    #         # Energy
    #         E = P / (param.gamma - 1.0) + 0.5 * rhoV * V
    #
    #         return k.stack([rho, rhoV, E], axis=2)


def generate_Riemann_Problem(param:Param,batch_size:int,k:K,minimum,maximal_jump):
    res = np.zeros([batch_size, param.nx])
    for i in range(batch_size):
        w_left = np.random.uniform(minimum,1)
        w_right = np.random.uniform(w_left*1.1,w_left*1.1+maximal_jump)
        res[i, :param.nx // 2] = w_left
        res[i, param.nx // 2:] = w_right
    return k.convert(res)



""" TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST  TEST TEST   """


def test_components():
    k = K("tf", 32)
    param = Param()
    genParam = GenParam(param, "???",64)
    generator = FuncGenerator(genParam, param, k)

    discount_loop = generator.generate_loop_dicontinuity()
    discount_changing0,discount_changing1 = generator.generate_two_changing_discontinuities_with_same_location()
    countinuity = generator.generate_fourier()

    fig, axs = plt.subplots(5, 3, sharey="all")
    for i in range(5):
        axs[i, 0].plot(discount_loop[i, :])
        axs[i, 1].plot(discount_changing0[i, :])
        axs[i, 1].plot(discount_changing1[i, :])

        axs[i, 2].plot(countinuity[i, :])

    axs[0, 0].set_title("loop")
    axs[0, 1].set_title("changing")
    axs[0, 2].set_title("continuity")

    plt.show()


#
# def test_generate_func_one(kind:str):
#     k = K("tf", 32)
#     param=Param()
#     genParam=GenerationParam(param,kind)
#     generator=FonctionGenerator(genParam,param,k)
#
#     res=generator.generate_func()
#     print(res.shape)
#
#     fig, axs = plt.subplots(5, 5)
#     axs = axs.flatten()
#     for i in range(25):
#         axs[i].plot(res[i,:])





def test_W_init():
    # k = K("tf", 32)
    param = Param()
    # genParam = GenParam(param, GenParam.kind_loop,64)
    # generator = FuncGenerator(genParam, param, k)
    W_loop = init_periodic(param,20)

    # genParam.kind=GenParam.kind_changing
    # generator = FuncGenerator(genParam, param, k)
    W_changing = init_non_periodic(param,20)

    W_sod = init_sod(param)

    fig, axs = plt.subplots(5, 3, sharey="all")
    for i in range(5):
        axs[i, 0].plot(W_loop[i, :, 0],label="rho loop")
        axs[i, 1].plot(W_changing[i, :, 0],label="rho changing")
        axs[i, 2].plot(W_sod[0, :, 0],label="rho sod")

    #plt.show()
    #
    # fig, axs = plt.subplots(5, 3, sharey="all")
    # for i in range(5):
    #     axs[i, 0].plot(W_loop[i, :, 1])
    #     axs[i, 1].plot(W_changing[i, :, 1])
    #     axs[i, 2].plot(W_sod[0, :, 1])
    #
    # axs[0, 0].set_title("rhoV loop")
    # axs[0, 1].set_title("rhoV changing")
    # axs[0, 2].set_title("rhoV sod")
    #
    # plt.show()

    #fig, axs = plt.subplots(5, 3, sharey="all")

    for i in range(5):
        axs[i, 0].plot(W_loop[i, :, 2],label="E loop")
        axs[i, 1].plot(W_changing[i, :, 2],label="E changing")
        axs[i, 2].plot(W_sod[0, :, 2],label="E sod")

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 2].legend()

    plt.show()


# test_generate_func()

if __name__ == "__main__":
    #test_components()
    #test_generate_func()
    test_W_init()
