# from tensorflow.python.types.core import Value
from Euler.param import Param, Projecter
from Euler.initial_conditions import *
import numpy as np
np.set_printoptions(precision=3, linewidth=100000)
pp = print


# pour comparer les performances
def compare(func_of_k):
    grid_nx = [100, 1000, 10000, 100000]
    grid_batch_size = [1, 32, 64, 128]

    def one_K(k):
        mat = np.zeros_float([len(grid_nx), len(grid_batch_size)])
        for i, nx in enumerate(grid_nx):
            for j, batch_size in enumerate(grid_batch_size):
                param = Param(nx, batch_size)
                t0 = time.time()
                func_of_k(k, param)
                mat[i, j] = time.time() - t0
        return mat
    mat_np = one_K(K("np", 32))
    mat_tf = one_K(K("tf", 64))
    df = pd.DataFrame(data=mat_np / mat_tf, columns=grid_batch_size, index=grid_nx)
    print(df)


def periodic_padding(W, k, pad):
    left = W[:, :pad, :]
    right = W[:, -pad:, :]
    return k.concatenate([right, W, left], axis=1)


def neumann_padding(W, k, pad):
    right_value = W[:, -1, :]
    left_value = W[:, 0, :]
    s = W.shape
    left_value_repeat = k.ones_float([s[0], pad, s[2]]) * left_value[:, k.newaxis, :]
    right_value_repeat = k.ones_float([s[0], pad, s[2]]) * right_value[:, k.newaxis, :]

    return k.concatenate([left_value_repeat, W, right_value_repeat], axis=1)


def reflexive_padding(W, k, pad):
    right_value = W[:, -1, :]
    left_value = W[:, 0, :]
    s = W.shape
    left_value_repeat = k.ones_float([s[0], pad, s[2]]) * left_value[:, k.newaxis, :]
    right_value_repeat = k.ones_float([s[0], pad, s[2]]) * right_value[:, k.newaxis, :]

    left = W[:, 1:pad + 1, :] - left_value_repeat
    right = W[:, -1 - pad:-1, :] - right_value_repeat
    left = -left[:, ::-1, :]
    right = -right[:, ::-1, :]
    left += left_value_repeat
    right += right_value_repeat

    return k.concatenate([left, W, right], axis=1)


class Var:
    def __init__(self, W: np.ndarray, param: Param, k: K):
        """  toutes les variables _XXX sont de dimensions nx+2 """
        self.W = W
        self.k = k
        self.param = param

        # la version prolongée de w
        if param.BC_solver == Param.BC_periodic:
            self.W_ = periodic_padding(W, k, 1)
        elif param.BC_solver == Param.BC_neumann:
            self.W_ = neumann_padding(W, k, 1)
        elif param.BC_solver == Param.BC_reflexive:
            self.W_ = reflexive_padding(W, k, 1)
        else:
            raise Exception("cette Boundary Condition pour le solver est inconnue:" + param.BC_solver)

        # des alias
        self.rho_ = self.W_[:, :, 0]  # density
        self.rhoV_ = self.W_[:, :, 1]  # moment
        self.E_ = self.W_[:, :, 2]  # energy

        self.V_ = self.rhoV_ / self.rho_

        self.rhoVV_ = self.rhoV_ * self.V_

        self.p_ = (param.gamma - 1) * (self.E_ - 1 / 2 * self.rhoVV_)  # pressure

        p_over_rho = self.p_ / self.rho_
        # pour éviter des Nan produite par de valeur négatives (résiduelle ?)
        p_over_rho = k.where_float(p_over_rho < 0., 0., p_over_rho)
        self.sound_ = k.sqrt(param.gamma * p_over_rho)  # sound_speed

        self.Flux_ = k.stack([self.rhoV_, self.rhoVV_ + self.p_, self.V_ * (self.E_ + self.p_)], axis=2)

    # nx+2

    def get_Flux(self):
        return self.Flux_

    def get(self, name, L_or_R):
        tensor = self.__dict__[name + "_"]
        if L_or_R == "L":
            return tensor[:, :-1]  # dim 2 ou 3
        elif L_or_R == "R":
            return tensor[:, 1:]  # dim 2 ou 3
        else:
            raise Exception("must be L or R")

    def get_augmentation(self):
        aug = self.k.stack([self.rho_, self.rhoV_, self.E_, self.V_, self.p_, self.sound_, self.Flux_[:, :, 1], self.Flux_[:, :, 2]], axis=2)
        assert self.param.augmentation_dim == aug.shape[2], "problème de shape: l'augmentation.shape=" + \
            str(aug.shape) + "si vous changer l'augmentation ci-dessus, il faut aussi changer le paramètres param.augmentation_dim"
        return aug[:, 1:-1, :]


def Flux_mean(var):
    return (var.get("Flux", "L") + var.get("Flux", "R")) * 0.5


def Flux_HLL(var: Var, param: Param, k: K):
    return Flux_HLLC(var, param, k)
    # vrai flux
    F_L = var.get("Flux", "L")
    F_R = var.get("Flux", "R")

    lamb_L = k.minimum(var.get("V", "L") - var.get("sound", "L"), var.get("V", "R") - var.get("sound", "R"))
    lamb_R = k.maximum(var.get("V", "L") + var.get("sound", "L"), var.get("V", "R") + var.get("sound", "R"))

    # loc_stab_loss = k.maximum(k.abs(lamb_L), k.abs(lamb_R))
    # stab_loss = k.max(loc_stab_loss)
    # max_stab_loss = param.CFL / param.dt_over_dx

    W_L = var.get("w", "L")
    W_R = var.get("w", "R")

    zone_1 = (lamb_L >= 0)[:, :, k.newaxis]
    zone_2 = (lamb_R <= 0)[:, :, k.newaxis]
    zone_3 = k.logical_not(k.logical_or(zone_1, zone_2))

    lamb_L = lamb_L[:, :, k.newaxis]
    lamb_R = lamb_R[:, :, k.newaxis]
    value_3 = ((lamb_R * F_L - lamb_L * F_R + lamb_L * lamb_R * (W_R - W_L)) / (lamb_R - lamb_L))

    numerical_Flux = k.where_float(zone_1, F_L, F_R)
    numerical_Flux = k.where_float(zone_3, value_3, numerical_Flux)

    return numerical_Flux


def Flux_HLL_LR(var_L: Var, var_R: Var, param: Param, k: K):
    return Flux_HLLC_LR(var_L, var_R, param, k)
    # vrai flux
    F_L = var_L.get("Flux", "L")
    F_R = var_R.get("Flux", "R")

    lamb_L = k.minimum(var_L.get("V", "L") - var_L.get("sound", "L"), var_R.get("V", "R") - var_R.get("sound", "R"))
    lamb_R = k.maximum(var_L.get("V", "L") + var_L.get("sound", "L"), var_R.get("V", "R") + var_R.get("sound", "R"))

    # loc_stab_loss = k.maximum(k.abs(lamb_L), k.abs(lamb_R))
    # stab_loss = k.max(loc_stab_loss)
    # max_stab_loss = param.CFL / param.dt_over_dx

    W_L = var_L.get("w", "L")
    W_R = var_R.get("w", "R")

    zone_1 = (lamb_L >= 0)[:, :, k.newaxis]
    zone_2 = (lamb_R <= 0)[:, :, k.newaxis]
    zone_3 = k.logical_not(k.logical_or(zone_1, zone_2))

    lamb_L = lamb_L[:, :, k.newaxis]
    lamb_R = lamb_R[:, :, k.newaxis]
    value_3 = ((lamb_R * F_L - lamb_L * F_R + lamb_L * lamb_R * (W_R - W_L)) / (lamb_R - lamb_L))

    numerical_Flux = k.where_float(zone_1, F_L, F_R)
    numerical_Flux = k.where_float(zone_3, value_3, numerical_Flux)

    return numerical_Flux


def HLLC_Wave_Speeds(Density_L, Density_R, Velocity_L, Velocity_R, Pressure_L, Pressure_R, Sound_L, Sound_R, param: Param, k: K):
    Pressure_Mean = (Pressure_L + Pressure_R) / 2
    Density_Mean = (Density_L + Density_R) / 2
    Sound_Mean = (Sound_L + Sound_R) / 2

    Pressure_Star = k.maximum(k.zeros_like(Pressure_Mean),
                              Pressure_Mean - Density_Mean * Sound_Mean * (Velocity_R - Velocity_L) / 2)

    condition_L = Pressure_Star > Pressure_L
    q_L = k.where_float(condition_L, k.sqrt(1 + (param.gamma + 1) / (2 * param.gamma) * (Pressure_Star / Pressure_L - 1)), k.ones_like(Pressure_Star))

    condition_R = Pressure_Star > Pressure_R
    q_R = k.where_float(condition_R, k.sqrt(1 + (param.gamma + 1) / (2 * param.gamma) * (Pressure_Star / Pressure_R - 1)), k.ones_like(Pressure_Star))

    return (Velocity_L - Sound_L * q_L,
            Velocity_R + Sound_R * q_R)


def Flux_HLLC(var: Var, param: Param, k: K):
    # vrai flux

    W_L = var.get("w", "L")
    W_R = var.get("w", "R")

    F_L = var.get("Flux", "L")
    F_R = var.get("Flux", "R")

    Density_L = W_L[:, :, 0]
    Density_R = W_R[:, :, 0]

    Momentum_L = W_L[:, :, 1]
    Momentum_R = W_R[:, :, 1]

    Velocity_L = var.get("V", "L")
    Velocity_R = var.get("V", "R")

    Pressure_L = var.get("p", "L")
    Pressure_R = var.get("p", "R")

    Sound_L = var.get("sound", "L")
    Sound_R = var.get("sound", "R")

    lamb_L, lamb_R = HLLC_Wave_Speeds(Density_L, Density_R, Velocity_L, Velocity_R, Pressure_L, Pressure_R, Sound_L, Sound_R, param, k)

    Lambda_Star = ((Pressure_R - Pressure_L
                  + Momentum_L * (lamb_L - Velocity_L)
                  - Momentum_R * (lamb_R - Velocity_R))
                  / (Density_L * (lamb_L - Velocity_L)
                   - Density_R * (lamb_R - Velocity_R)))

    Pressure_Star = Pressure_L + Density_L * (lamb_L - Velocity_L) * (Lambda_Star - Velocity_L)

    Density_L_Star = Density_L * (lamb_L - Velocity_L) / (lamb_L - Lambda_Star)
    W_L_Star_0 = Density_L_Star
    W_L_Star_1 = Density_L_Star * Lambda_Star
    W_L_Star_2 = Pressure_Star / (param.gamma - 1) + Density_L_Star * Lambda_Star**2 / 2
    W_L_Star = k.stack((W_L_Star_0, W_L_Star_1, W_L_Star_2), axis=2)

    Density_R_Star = Density_R * (lamb_R - Velocity_R) / (lamb_R - Lambda_Star)
    W_R_Star_0 = Density_R_Star
    W_R_Star_1 = Density_R_Star * Lambda_Star
    W_R_Star_2 = Pressure_Star / (param.gamma - 1) + Density_R_Star * Lambda_Star**2 / 2
    W_R_Star = k.stack((W_R_Star_0, W_R_Star_1, W_R_Star_2), axis=2)

    lamb_L = lamb_L[:, :, np.newaxis]
    Lambda_Star = Lambda_Star[:, :, np.newaxis]
    lamb_R = lamb_R[:, :, np.newaxis]

    F_L_Star = F_L + lamb_L * (W_L_Star - W_L)
    F_R_Star = F_R + lamb_R * (W_R_Star - W_R)

    zone_1 = (lamb_L >= 0)
    zone_2 = k.logical_and((Lambda_Star >= 0), (lamb_L < 0))
    zone_3 = k.logical_and((lamb_R >= 0), (Lambda_Star < 0))
    zone_4 = (lamb_R < 0)

    numerical_Flux = k.where_float(zone_1, F_L, F_L_Star)
    numerical_Flux = k.where_float(zone_3, F_R_Star, numerical_Flux)
    numerical_Flux = k.where_float(zone_4, F_R, numerical_Flux)

    return numerical_Flux


def Flux_HLLC_LR(var_L: Var, var_R: Var, param: Param, k: K):
    # vrai flux

    W_L = var_L.get("w", "L")
    W_R = var_R.get("w", "R")

    F_L = var_L.get("Flux", "L")
    F_R = var_R.get("Flux", "R")

    Density_L = W_L[:, :, 0]
    Density_R = W_R[:, :, 0]

    Momentum_L = W_L[:, :, 1]
    Momentum_R = W_R[:, :, 1]

    Velocity_L = var_L.get("V", "L")
    Velocity_R = var_R.get("V", "R")

    Pressure_L = var_L.get("p", "L")
    Pressure_R = var_R.get("p", "R")

    Sound_L = var_L.get("sound", "L")
    Sound_R = var_R.get("sound", "R")

    lamb_L, lamb_R = HLLC_Wave_Speeds(Density_L, Density_R, Velocity_L, Velocity_R, Pressure_L, Pressure_R, Sound_L, Sound_R, param, k)

    Lambda_Star = ((Pressure_R - Pressure_L
                  + Momentum_L * (lamb_L - Velocity_L)
                  - Momentum_R * (lamb_R - Velocity_R))
                  / (Density_L * (lamb_L - Velocity_L)
                   - Density_R * (lamb_R - Velocity_R)))

    Pressure_Star = Pressure_L + Density_L * (lamb_L - Velocity_L) * (Lambda_Star - Velocity_L)

    Density_L_Star = Density_L * (lamb_L - Velocity_L) / (lamb_L - Lambda_Star)
    W_L_Star_0 = Density_L_Star
    W_L_Star_1 = Density_L_Star * Lambda_Star
    W_L_Star_2 = Pressure_Star / (param.gamma - 1) + Density_L_Star * Lambda_Star**2 / 2
    W_L_Star = k.stack((W_L_Star_0, W_L_Star_1, W_L_Star_2), axis=2)

    Density_R_Star = Density_R * (lamb_R - Velocity_R) / (lamb_R - Lambda_Star)
    W_R_Star_0 = Density_R_Star
    W_R_Star_1 = Density_R_Star * Lambda_Star
    W_R_Star_2 = Pressure_Star / (param.gamma - 1) + Density_R_Star * Lambda_Star**2 / 2
    W_R_Star = k.stack((W_R_Star_0, W_R_Star_1, W_R_Star_2), axis=2)

    lamb_L = lamb_L[:, :, np.newaxis]
    Lambda_Star = Lambda_Star[:, :, np.newaxis]
    lamb_R = lamb_R[:, :, np.newaxis]

    F_L_Star = F_L + lamb_L * (W_L_Star - W_L)
    F_R_Star = F_R + lamb_R * (W_R_Star - W_R)

    zone_1 = (lamb_L >= 0)
    zone_2 = k.logical_and((Lambda_Star >= 0), (lamb_L < 0))
    zone_3 = k.logical_and((lamb_R >= 0), (Lambda_Star < 0))
    zone_4 = (lamb_R < 0)

    numerical_Flux = k.where_float(zone_1, F_L, F_L_Star)
    numerical_Flux = k.where_float(zone_3, F_R_Star, numerical_Flux)
    numerical_Flux = k.where_float(zone_4, F_R, numerical_Flux)

    return numerical_Flux


def minmod(a, b, k: K):
    """ minmod limiter """
    c1 = k.logical_and(a > 0, b > 0)
    c2 = k.logical_and(a < 0, b < 0)

    limiter = k.where_float(c1, k.minimum(a, b), k.zeros_like(a))
    limiter = k.where_float(c2, k.maximum(a, b), limiter)

    return limiter


def SST_viscous(a, b, k: K):
    """ order 3 limiter without smoothness detection [ Schmidtmann Seibold Torrilhon 2015 ] """

    # zeros_N = k.zeros_float(a.shape)
    # limiter = np.copy(zeros_N)

    positive_a = a > 0
    positive_b = b > 0
    zeros = k.zeros_like(a)

    c1 = np.logical_and(positive_a, positive_b)
    c2 = np.logical_and(np.logical_not(positive_a), positive_b)
    c3 = np.logical_and(positive_a, np.logical_not(positive_b))
    c4 = np.logical_and(np.logical_not(positive_a), np.logical_not(positive_b))

    limiter = k.where_float(c1, k.maximum(zeros, k.minimum(k.minimum(2 * a, 3 / 2 * b), (a + 2 * b) / 3)), zeros)
    limiter = k.where_float(c2, k.maximum(zeros, k.minimum(- a, (a + 2 * b) / 3)), limiter)
    limiter = k.where_float(c3, k.minimum(zeros, k.maximum(- a, (a + 2 * b) / 3)), limiter)
    limiter = k.where_float(c4, k.minimum(zeros, k.maximum(k.maximum(2 * a, 3 / 2 * b), (a + 2 * b) / 3)), limiter)

    return limiter


def compute_solutions_coarse(param, nb_t, W_init, k: K, HLLC):
    res = []
    W = W_init  # inutile de faire une copie, elle est faites dans le Var_burger() au moment du padding
    for t in range(nb_t):
        res.append(W)
        var = Var(W, param, k)
        if HLLC:
            Fnum = Flux_HLLC(var, param, k)
        else:
            Fnum = Flux_HLL(var, param, k)
        dt_over_dx = param.dt_over_dx_coarse
        W = W - dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])

    return k.stack(res)


def compute_solutions_order1(param, nb_t, W_init, k: K, HLLC):
    """ res=k.zeros_float((nb_t,) + W_init.shape) ne fonctionne pas à cause de: EagerTensor' object does not support item assignment  """
    res = []
    W = W_init  # inutile de faire une copie, elle est faites dans le Var_burger() au moment du padding
    for t in range(nb_t):
        res.append(W)
        var = Var(W, param, k)
        if HLLC:
            Fnum = Flux_HLLC(var, param, k)
        else:
            Fnum = Flux_HLL(var, param, k)
        dt_over_dx = param.dt_over_dx
        W = W - dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])

    return k.stack(res)



def time_step_order2(W, param, k: K, HLLC):

    # la version prolongée de w
    if param.BC_solver == Param.BC_periodic:
        W_ = periodic_padding(W, k, 1)
    elif param.BC_solver == Param.BC_neumann:
        W_ = neumann_padding(W, k, 1)
    elif param.BC_solver == Param.BC_reflexive:
        W_ = reflexive_padding(W, k, 1)
    else:
        raise Exception("cette Boundary Condition pour le solver est inconnue:" + param.BC_solver)

    slopes = W_[:, 1:, :] - W_[:, :-1, :]

    limited_slopes = minmod(slopes[:, :-1, :], slopes[:, 1:, :], k)

    var_recon_L = Var(W - limited_slopes / 2, param, k)
    var_recon_R = Var(W + limited_slopes / 2, param, k)

    if HLLC:
        Fnum = Flux_HLLC_LR(var_recon_R, var_recon_L, param, k)
    else:
        Fnum = Flux_HLL_LR(var_recon_R, var_recon_L, param, k)

    dt_over_dx = param.dt_over_dx
    return W - dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])




def compute_solutions_order2(param, nb_t, W_init, k: K, HLLC):
    res = []

    W = W_init  # inutile de faire une copie, elle est faites dans le Var_burger() au moment du padding

    for t in range(nb_t):
        res.append(W)
        W_1 = time_step_order2(res[-1], param, k, HLLC)
        W_2 = time_step_order2(W_1, param, k, HLLC)
        W = (res[-1] + W_2) / 2

    return k.stack(res)


def time_step_order3(W, param, k: K, HLLC):

    # la version prolongée de w
    if param.BC_solver == Param.BC_periodic:
        W_ = periodic_padding(W, k, 1)
    elif param.BC_solver == Param.BC_neumann:
        W_ = neumann_padding(W, k, 1)
    elif param.BC_solver == Param.BC_reflexive:
        W_ = reflexive_padding(W, k, 1)
    else:
        raise Exception("cette Boundary Condition pour le solver est inconnue:" + param.BC_solver)

    slopes = W_[:, 1:, :] - W_[:, :-1, :]

    limited_slopes_L = SST_viscous(slopes[:, 1:, :], slopes[:, :-1, :], k)
    limited_slopes_R = SST_viscous(slopes[:, :-1, :], slopes[:, 1:, :], k)

    var_recon_L = Var(W - limited_slopes_L / 2, param, k)
    var_recon_R = Var(W + limited_slopes_R / 2, param, k)

    if HLLC:
        Fnum = Flux_HLLC_LR(var_recon_R, var_recon_L, param, k)
    else:
        Fnum = Flux_HLL_LR(var_recon_R, var_recon_L, param, k)

    dt_over_dx = param.dt_over_dx
    return W - dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])


def compute_solutions_order3(param, nb_t, W_init, k: K, HLLC):
    res = []

    W = W_init  # inutile de faire une copie, elle est faites dans le Var_burger() au moment du padding

    for t in range(nb_t):
        res.append(W)
        W_1 = time_step_order3(res[-1], param, k, HLLC)
        W_2 = time_step_order3(W_1, param, k, HLLC)
        W_2 = (3 * res[-1] + 1 * W_2) / 4
        W_3 = time_step_order3(W_2, param, k, HLLC)
        W = (1 * res[-1] + 2 * W_3) / 3

    return k.stack(res)


def compute_solutions(param, nb_t, W_init, is_coarse, k: K, order, HLLC):
    """"""
    if is_coarse:
        assert W_init.shape[1] == param.nx_coarse
        return compute_solutions_coarse(param, nb_t, W_init, k, HLLC=False)
    else:
        assert W_init.shape[1] == param.nx
        if order == 1:
            return compute_solutions_order1(param, nb_t, W_init, k, HLLC)
        elif order == 2:
            return compute_solutions_order2(param, nb_t, W_init, k, HLLC)
        elif order == 3:
            return compute_solutions_order3(param, nb_t, W_init, k, HLLC)
        else:
            raise ValueError("order > 3 is not supported in compute_solutions")


def compute_Fnum_order1(W, param, k, HLLC):

    var = Var(W, param, k)
    if HLLC:
        return Flux_HLLC(var, param, k)
    else:
        return Flux_HLL(var, param, k)


def compute_dFnum_order1(W, param, k, HLLC):
    """ first-order in time """

    Fnum = compute_Fnum_order1(W, param, k, HLLC)
    return - (Fnum[:, 1:, :] - Fnum[:, :-1, :])


def compute_Fnum_order2(W, param, k, HLLC):

    # la version prolongée de w
    if param.BC_solver == Param.BC_periodic:
        W_ = periodic_padding(W, k, 1)
    elif param.BC_solver == Param.BC_neumann:
        W_ = neumann_padding(W, k, 1)
    elif param.BC_solver == Param.BC_reflexive:
        W_ = reflexive_padding(W, k, 1)
    else:
        raise Exception("cette Boundary Condition pour le solver est inconnue:" + param.BC_solver)

    slopes = W_[:, 1:, :] - W_[:, :-1, :]

    limited_slopes = minmod(slopes[:, :-1, :], slopes[:, 1:, :], k)

    var_recon_L = Var(W - limited_slopes / 2, param, k)
    var_recon_R = Var(W + limited_slopes / 2, param, k)

    if HLLC:
        return Flux_HLLC_LR(var_recon_R, var_recon_L, param, k)
    else:
        return Flux_HLL_LR(var_recon_R, var_recon_L, param, k)


def compute_dFnum_order2(W, param, k, HLLC):
    """ second-order in time """

    Fnum = compute_Fnum_order2(W, param, k, HLLC)
    W_1 = W - param.dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])

    Fnum = compute_Fnum_order2(W_1, param, k, HLLC)
    W_2 = W_1 - param.dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])

    return ((W + W_2) / 2 - W) / param.dt_over_dx


def compute_Fnum_order3(W, param, k, HLLC):

    # la version prolongée de w
    if param.BC_solver == Param.BC_periodic:
        W_ = periodic_padding(W, k, 1)
    elif param.BC_solver == Param.BC_neumann:
        W_ = neumann_padding(W, k, 1)
    elif param.BC_solver == Param.BC_reflexive:
        W_ = reflexive_padding(W, k, 1)
    else:
        raise Exception("cette Boundary Condition pour le solver est inconnue:" + param.BC_solver)

    slopes = W_[:, 1:, :] - W_[:, :-1, :]

    limited_slopes_L = SST_viscous(slopes[:, 1:, :], slopes[:, :-1, :], k)
    limited_slopes_R = SST_viscous(slopes[:, :-1, :], slopes[:, 1:, :], k)

    var_recon_L = Var(W - limited_slopes_L / 2, param, k)
    var_recon_R = Var(W + limited_slopes_R / 2, param, k)

    if HLLC:
        return Flux_HLLC_LR(var_recon_R, var_recon_L, param, k)
    else:
        return Flux_HLL_LR(var_recon_R, var_recon_L, param, k)


def compute_dFnum_order3(W, param, k, HLLC):
    """ third-order in time """

    Fnum = compute_Fnum_order3(W, param, k, HLLC)
    W_1 = W - param.dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])

    Fnum = compute_Fnum_order3(W_1, param, k, HLLC)
    W_2 = W_1 - param.dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])
    W_2 = (3 * W + 1 * W_2) / 4

    Fnum = compute_Fnum_order3(W_2, param, k, HLLC)
    W_3 = W_2 - param.dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])

    return ((W + 2 * W_3) / 3 - W) / param.dt_over_dx


def compute_fine_solutions(genParam: GenParam, param: Param, nb_t, order, HLLC, same_discontinuity_locations=False, compute_coarse_solution=False):
    k = K("tf", 64)
    generator = FuncGenerator(genParam, param, k)
    W_init = generator.init_W(same_discontinuity_locations)
    ti = time.time()

    res_W = [None] * nb_t  #
    res_dFnum = [None] * nb_t

    if compute_coarse_solution:
        projecter = Projecter(param.nx_ratio, 64)
        W_coarse = projecter.projection(W_init)

    W = W_init

    if order > 3:
        raise ValueError("order > 3 is not supported in compute_fine_solutions")

    for t in range(nb_t):

        res_W[t] = W

        if order == 1:
            # Fnum = compute_Fnum_order1(w, param, k, HLLC)
            res_dFnum[t] = compute_dFnum_order1(W, param, k, HLLC)
        elif order == 2:
            # Fnum = compute_Fnum_order2(w, param, k, HLLC)
            res_dFnum[t] = compute_dFnum_order2(W, param, k, HLLC)
        elif order == 3:
            # Fnum = compute_Fnum_order3(w, param, k, HLLC)
            res_dFnum[t] = compute_dFnum_order3(W, param, k, HLLC)

        dt_over_dx = param.dt_over_dx
        W = res_W[t] + dt_over_dx * res_dFnum[t]

    if compute_coarse_solution:
        for _ in range(nb_t):
            var = Var(W_coarse, param, k)
            Fnum_HLL = Flux_HLL(var, param, k)
            dFnum_HLL = (Fnum_HLL[:, 1:, :] - Fnum_HLL[:, :-1, :])

            dFnum = dFnum_HLL

            W_coarse = W_coarse - param.dt_over_dx_coarse * dFnum

    W, Y = k.stack(res_W), k.stack(res_dFnum)

    #     res_Fnum.append(Fnum)

    #     dt_over_dx = param.dt_over_dx
    #     w = w - dt_over_dx * (Fnum[:, 1:, :] - Fnum[:, :-1, :])

    # res_Fnum = k.stack(res_Fnum)
    # res_dFnum = res_Fnum[:, :, :-1, :] - res_Fnum[:, :, 1:, :]

    # w, Y = k.stack(res_W), res_dFnum

    W = tf.cast(W, tf.float32)
    Y = tf.cast(Y, tf.float32)

    print("durée du calcul de la solution fine:", time.time() - ti, end="")

    if compute_coarse_solution:
        return W, Y, W_coarse
    else:
        return W, Y





def compare_with_projection_one_kind(kind):
    k = K("np", 32)

    param = Param(6000, batch_size=2, BC_solver=Param.BC_reflexive)

    genParam = GenParam(param, kind)
    generator = FuncGenerator(genParam, param, k)

    W = generator.init_W()
    nb_t = 5000

    W_coarse = Projecter(param.nx_ratio, 32).projection(W)
    if k.kind == "np":
        W_coarse = k.convert(W_coarse)

    res = compute_solutions(param, nb_t, W, False, k)
    res_coarse = compute_solutions(param, nb_t, W_coarse, True, k)

    fig, ax = plt.subplots(3, 2)
    fig.suptitle(kind)

    for t in range(0, nb_t, nb_t // 10):
        color = "r" if t == 0 else "k"
        alpha = 1 if t == 0 else t / nb_t
        ax[0, 0].set_title("rho")
        ax[0, 0].plot(res[t, 0, :, 0], color, alpha=alpha)
        ax[0, 1].plot(res_coarse[t, 0, :, 0], color, alpha=alpha)
        ax[1, 0].set_title("rhoV")
        ax[1, 0].plot(res[t, 0, :, 1], color, alpha=alpha)
        ax[1, 1].plot(res_coarse[t, 0, :, 1], color, alpha=alpha)
        ax[2, 0].set_title("E")
        ax[2, 0].plot(res[t, 0, :, 2], color, alpha=alpha)
        ax[2, 1].plot(res_coarse[t, 0, :, 2], color, alpha=alpha)

    fig.tight_layout()

    plt.show()


def test_the_3_paddings():
    k = K('tf', 32)
    W = k.arange_float(0, 9)[k.newaxis, :, k.newaxis]
    W_neumann = neumann_padding(W, k, 2)
    W_periodic = periodic_padding(W, k, 2)
    W_reflexive = reflexive_padding(W, k, 2)
    assert W_neumann.shape == W_reflexive.shape == W_periodic.shape
    print(W_neumann.shape)
    plt.plot(W_neumann[0, :, 0], label="neumann")
    plt.plot(W_periodic[0, :, 0], label="periodic")
    plt.plot(W_reflexive[0, :, 0], label="reflexive")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_the_3_paddings()
    compare_with_projection_one_kind(GenParam.kind_sod)
    compare_with_projection_one_kind(GenParam.kind_loop)
    compare_with_projection_one_kind(GenParam.kind_changing)
