import pandas as pd
from Euler.initial_conditions_burger import FuncGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from Euler.param import Param, Projecter
import numpy as np
from Euler.neural_network import Model
pp = print


@tf.function
def compute_solutions_nan_filtred(param: Param, nb_t, w_init, order: int, model, addi: bool):
    res = compute_solutions_accelerate(param, nb_t, w_init, order, model, addi)
    res_sum = tf.reduce_sum(res, axis=[0, 2, 3])
    non_nan_line = tf.logical_not(tf.math.is_nan(res_sum))
    res_non_nan = tf.boolean_mask(res, non_nan_line, axis=1)
    return res_non_nan


def one_time_step_with_slopes(param: Param, dt_over_dx: float, w: tf.Tensor, order: int, model, addi: bool) -> tf.Tensor:

    def time_step(w) -> tf.Tensor:
        if order == 1:
            Fnum = flux_with_diffu_order1(param, w, model, addi)
        else:
            Fnum, slopes, slopes_minmod, slopes_o2 = flux_with_diffu_order2(param, w, model, addi, get_slopes=True)
        dFnum = Fnum[:, 1:, :] - Fnum[:, :-1, :]
        return w - dt_over_dx * dFnum, slopes[:, 1:-1, :], slopes_minmod[:, 1:-1, :], slopes_o2[:, 1:-1, :]

    w_t1, slopes_t1, slopes_minmod_t1, slopes_o2_t1 = time_step(w)
    w_t2, slopes_t2, slopes_minmod_t2, slopes_o2_t2 = time_step(w_t1)
    return (w + w_t2) / 2, (slopes_t1 + slopes_t2) / 2, (slopes_minmod_t1 + slopes_minmod_t2) / 2, (slopes_o2_t1 + slopes_o2_t2) / 2


def one_time_step(param: Param, dt_over_dx: float, w: tf.Tensor, order: int, model, addi: bool) -> tf.Tensor:

    def time_step(w) -> tf.Tensor:
        if order == 1:
            Fnum = flux_with_diffu_order1(param, w, model, addi)
        else:
            Fnum = flux_with_diffu_order2(param, w, model, addi)
        dFnum = Fnum[:, 1:, :] - Fnum[:, :-1, :]
        return w - dt_over_dx * dFnum

    w_t1 = time_step(w)
    w_t2 = time_step(w_t1)
    return (w + w_t2) / 2


@tf.function
def compute_solutions_accelerate(param: Param, nb_t, w_init, order: int, model, addi: bool):

    (b, nx, d) = w_init.shape

    if nx == param.nx_coarse:
        dt_over_dx = param.dt_over_dx_coarse
    elif nx == param.nx:
        dt_over_dx = param.dt_over_dx
    else:
        raise Exception("l'entrée w_init n'a pas la bonne shape")

    res = tf.TensorArray(tf.float32, size=nb_t, element_shape=[b, nx, d], dynamic_size=False, clear_after_read=True)

    w = w_init
    for t in tf.range(nb_t):
        res = res.write(t, w)
        w = one_time_step(param, dt_over_dx, w, order, model, addi)

    return res.stack()


@tf.function
def compute_solutions_with_slopes(param: Param, nb_t, w_init, order: int, model, addi: bool):

    (b, nx, d) = w_init.shape

    if nx == param.nx_coarse:
        dt_over_dx = param.dt_over_dx_coarse
    elif nx == param.nx:
        dt_over_dx = param.dt_over_dx
    else:
        raise Exception("l'entrée w_init n'a pas la bonne shape")

    res = tf.TensorArray(tf.float32, size=nb_t, element_shape=[b, nx, d], dynamic_size=False, clear_after_read=True)
    res_slopes = tf.TensorArray(tf.float32, size=nb_t, element_shape=[b, nx, d], dynamic_size=False, clear_after_read=True)
    res_slopes_minmod = tf.TensorArray(tf.float32, size=nb_t, element_shape=[b, nx, d], dynamic_size=False, clear_after_read=True)
    res_slopes_o2 = tf.TensorArray(tf.float32, size=nb_t, element_shape=[b, nx, d], dynamic_size=False, clear_after_read=True)

    w = w_init
    slopes = tf.zeros_like(w_init)
    slopes_minmod = tf.zeros_like(w_init)
    slopes_o2 = tf.zeros_like(w_init)
    for t in tf.range(nb_t):
        res = res.write(t, w)
        if model is not None:
            res_slopes = res_slopes.write(t, slopes)
            res_slopes_minmod = res_slopes_minmod.write(t, slopes_minmod)
            res_slopes_o2 = res_slopes_o2.write(t, slopes_o2)
            w, slopes, slopes_minmod, slopes_o2 = one_time_step_with_slopes(param, dt_over_dx, w, order, model, addi)
        else:
            w = one_time_step(param, dt_over_dx, w, order, model, addi)

    if model is not None:
        return res.stack(), res_slopes.stack(), res_slopes_minmod.stack(), res_slopes_o2.stack()
    else:
        return res.stack()


# fonction utile pour les 2 suivantes
def _flux_with_diffu(param: Param, w_moins, w_plus, D):
    F_mean = (param.F_fn(w_moins) + param.F_fn(w_plus)) / 2
    dw = w_plus - w_moins
    return F_mean - dw * D


def flux_with_diffu_order1(param: Param, w: tf.Tensor, model_D: Model, addi: bool):

    w_ = pad_w(w, param, 1)  # nx+2
    w_a, w_b = w_[:, :-1, :], w_[:, 1:, :]  # nx+1
    dw = w_b - w_a
    F_mean = (param.F_fn(w_a) + param.F_fn(w_b)) / 2

    if model_D is not None:
        # shrinkage impair
        p = (model_D.shrinkage + 1) // 2
        W = tf.concat([w, param.F_fn(w)], axis=2)  # pour eurler on pourrait mieux enrichier...
        W__ = pad_w(W, param, p)  # nx+2p
        D_model = model_D.call(W__)  # nx+2p-shr = nx + shr+1 -shr = nx+1
        if addi:
            D = D_model + param.D_fn_rusanov(w_a, w_b)  # nx+1
        else:
            D = D_model
    else:
        D = param.D_fn_rusanov(w_a, w_b)  # nx+1

    return F_mean - dw * D  # nx+1


def flux_with_diffu_order2(param: Param, w: tf.Tensor, model_L: Model, addi: bool, get_slopes=False):

    # il faut un shrinkage pair, ainsi 2p=shrinkage
    p = model_L.shrinkage // 2 if model_L is not None else 0
    # pour économiser une recopie des données, on fait un seul padding large (pour le modèle), on récupère ensuite la partie centrale (pour le calcul classique)
    w = pad_w(w, param, 2 + p)  # nx+4+2p
    dw = (w[:, 1:, :] - w[:, :-1, :]) / param.dx  # nx+3 +2p

    if model_L is not None:
        dW = tf.concat([dw[:, :-1, :], dw[:, 1:, :]], axis=2)  # nx+2 +2p
        L_model = param.dx / 2 * model_L.call(dW)  # nx+2+2p-shr = nx + 2

        w = w[:, p:-p, :]  # nx+4
        dw = dw[:, p:-p, :]  # nx+3

        if addi:
            L = L_model + param.dx / 2 * minmod(dw[:, :-1, :], dw[:, 1:, :])  # nx+2
        else:
            L = L_model
    else:
        L = param.dx / 2 * minmod(dw[:, :-1, :], dw[:, 1:, :])  # nx+2

    w_L = w[:, 1:-1, :] - L  # nx+2
    w_R = w[:, 1:-1, :] + L  # nx+2

    w_a, w_b = w_R[:, :-1, :], w_L[:, 1:, :]  # nx+1

    F_mean = (param.F_fn(w_a) + param.F_fn(w_b)) / 2
    D = param.D_fn_rusanov(w_a, w_b)  # nx+1

    if get_slopes and model_L is not None:
        return F_mean - D * (w_b - w_a), L_model, param.dx / 2 * minmod(dw[:, :-1, :], dw[:, 1:, :]), param.dx / 2 * (dw[:, :-1, :] + dw[:, 1:, :]) / 2  # nx+1
    else:
        return F_mean - D * (w_b - w_a)  # nx+1


def flux_with_diffu_order2_old(param: Param, w: tf.Tensor, model_L: Model):

    w = pad_w(w, param, 2)  # nx+4
    dw = (w[:, :-1, :] - w[:, 1:, :]) / param.dx  # nx+3
    L = param.dx / 2 * minmod(dw[:, :-1, :], dw[:, 1:, :])  # nx+2

    print("L", L.shape)
    if model_L is not None:
        # shrinkage paire
        p = model_L.shrinkage // 2
        w_ = pad_w(dw, param, p)  # nx+3+2p
        dw_ = (w_[:, :-1, :] - w_[:, 1:, :]) / param.dx  # nx+3+2p-1
        print("dw_", dw_.shape)
        L += model_L.call(dw_)  # nx+3+2p-1-shr = nx +3 +shr -1-shr=nx+2

    w_L = w[:, 1:-1, :] - L  # nx+2
    w_R = w[:, 1:-1, :] + L  # nx+2

    w_a, w_b = w_R[:, :-1, :], w_L[:, 1:, :]  # nx+1

    dw = w_b - w_a
    F_mean = (param.F_fn(w_a) + param.F_fn(w_b)) / 2
    D = param.D_fn_rusanov(w_a, w_b)  # nx+1

    return F_mean - dw * D  # nx+1


def minmod(a, b):
    """ minmod limiter """
    c1 = tf.logical_and(a > 0, b > 0)
    c2 = tf.logical_and(a < 0, b < 0)

    limiter = tf.where(c1, tf.minimum(a, b), tf.zeros_like(a))
    limiter = tf.where(c2, tf.maximum(a, b), limiter)

    return limiter


def pad_w(W, param: Param, p: int):
    if param.BC == Param.BC_periodic:
        return periodic_padding(W, p)
    elif param.BC == Param.BC_neumann:
        return neumann_padding(W, p)
    elif param.BC == Param.BC_reflexive and param.problem == "burger":
        return reflexive_padding(W, p, False)
    else:
        raise Exception("cette Boundary Condition pour le solver est inconnue:" + param.BC)


def periodic_padding(W, pad: int):
    left = W[:, :pad, :]
    right = W[:, -pad:, :]
    return tf.concat([right, W, left], axis=1)


def neumann_padding(W, pad):
    right_value = W[:, -1, :]
    left_value = W[:, 0, :]
    s = W.shape
    left_value_repeat = tf.ones([s[0], pad, s[2]]) * left_value[:, tf.newaxis, :]
    right_value_repeat = tf.ones([s[0], pad, s[2]]) * right_value[:, tf.newaxis, :]

    return tf.concat([left_value_repeat, W, right_value_repeat], axis=1)


def activation_for_relexive():
    x = tf.linspace(-3, 3, 100)
    y = tf.nn.elu(x - 1) + 1
    z = tf.nn.relu(x)
    plt.plot(x, y)
    plt.plot(x, z)
    plt.show()


# uniquement pour Euler
def make_positive_channels_0_2(A):

    A0, A1, A2 = A[:, :, 0], A[:, :, 1], A[:, :, 2]
    A0 = tf.nn.elu(A0 - 1) + 1
    A2 = tf.nn.elu(A2 - 1) + 1
    res = tf.stack([A0, A1, A2], axis=2)
    return res


def reflexive_padding(W: tf.Tensor, pad: int, make_positive: bool):

    right_value = W[:, -1, :]
    left_value = W[:, 0, :]
    s = W.shape
    left_value_repeat = tf.ones([s[0], pad, s[2]]) * left_value[:, tf.newaxis, :]
    right_value_repeat = tf.ones([s[0], pad, s[2]]) * right_value[:, tf.newaxis, :]

    left = W[:, 1:pad + 1, :] - left_value_repeat
    right = W[:, -1 - pad:-1, :] - right_value_repeat
    left = -left[:, ::-1, :]
    right = -right[:, ::-1, :]
    left += left_value_repeat
    right += right_value_repeat

    if make_positive and make_positive:
        left = make_positive_channels_0_2(left)
        right = make_positive_channels_0_2(right)

    return tf.concat([left, W, right], axis=1)


def test_the_3_paddings():
    x = tf.cast(np.linspace(0., 1, 110, endpoint=False), tf.float32)
    y = tf.sin(x * 2 * np.pi) + (2 * x + 1)
    W = tf.stack([y, y, y], axis=1)
    W = W[tf.newaxis, :, :]

    pad = 100
    W_neumann = neumann_padding(W, pad)
    W_periodic = periodic_padding(W, pad)
    W_reflexive = reflexive_padding(W, pad, False)
    W_reflexive_pos = reflexive_padding(W, pad, True)

    assert W_neumann.shape == W_reflexive.shape == W_periodic.shape
    plt.plot(W_neumann[0, :, 0], label="neumann")
    plt.plot(W_periodic[0, :, 0], label="periodic")
    plt.plot(W_reflexive[0, :, 0], label="reflexive")
    plt.plot(W_reflexive_pos[0, :, 0], label="reflexive_pos")
    plt.plot(np.zeros_like(x), color="k")
    plt.legend()
    plt.show()


def score_with_classic():
    param = Param(nx=1000, nx_ratio=5, problem="burger")
    funcGen = FuncGenerator(param, 200, False)

    # w_init=tf.sin(2*np.pi*param.xs)[tf.newaxis,:,tf.newaxis]

    w_init = funcGen()
    projecter = Projecter(param.nx_ratio)
    w_init_coarse = projecter.projection_3D(w_init)
    nb_t = 1000

    res_1 = compute_solutions_nan_filtred(param, nb_t, w_init, 1, None, False)
    print("res_1", res_1.shape, tf.reduce_mean(tf.abs(res_1)))
    print(f"{res_1.shape[1]}/{w_init.shape[0]} non-nan solution")

    res_coarse_1 = compute_solutions_nan_filtred(param, nb_t, w_init_coarse, 1, None, False)
    print("res_coarse_1", res_coarse_1.shape, tf.reduce_mean(tf.abs(res_coarse_1)))
    print(f"{res_coarse_1.shape[1]}/{w_init_coarse.shape[0]} non-nan solution")

    res_2 = compute_solutions_nan_filtred(param, nb_t, w_init, 2, None, False)  # pas de modèle pour la solution fine
    print("res_2", res_2.shape, tf.reduce_mean(tf.abs(res_2)))
    print(f"{res_2.shape[1]}/{w_init.shape[0]} non-nan solution")

    res_coarse_2 = compute_solutions_nan_filtred(param, nb_t, w_init_coarse, 2, None, False)
    print("res_coarse_2", res_coarse_2.shape, tf.reduce_mean(tf.abs(res_coarse_2)))
    print(f"{res_coarse_2.shape[1]}/{w_init_coarse.shape[0]} non-nan solution")

    df = pd.DataFrame()

    ref = projecter.projection_4D(res_2)

    def score(a):
        return tf.reduce_mean(tf.abs(a - ref)).numpy()

    df.loc["order 1", "fine"] = score(projecter.projection_4D(res_1))
    df.loc["order 1", "coarse"] = score(res_coarse_1)
    df.loc["order 2", "fine"] = score(projecter.projection_4D(res_2))
    df.loc["order 2", "coarse"] = score(res_coarse_2)

    print(df)

    fig, ax = plt.subplots(2, 2)
    for t in range(0, nb_t, nb_t // 10):
        color = "r" if t == 0 else "k"
        alpha = 1 if t == 0 else t / nb_t
        ax[0, 0].set_title("order 1")
        ax[0, 0].plot(res_1[t, 0, :, 0], color, alpha=alpha)

        ax[0, 1].set_title("order 1 coarse")
        ax[0, 1].plot(res_coarse_1[t, 0, :, 0], color, alpha=alpha)

        ax[1, 0].set_title("order 2")
        ax[1, 0].plot(res_2[t, 0, :, 0], color, alpha=alpha)

        ax[1, 1].set_title("order 2 coarse")
        ax[1, 1].plot(res_coarse_2[t, 0, :, 0], color, alpha=alpha)

    fig.tight_layout()
    plt.show()


def test_compute_solution(with_model: bool, addi: bool):
    param = Param(nx=1000, problem="burger")
    funcGen = FuncGenerator(param, 10, False)

    # w_init=tf.sin(2*np.pi*param.xs[tf.newaxis,:,tf.newaxis])
    w_init = funcGen()
    w_init_coarse = Projecter(param.nx_ratio).projection_3D(w_init)

    nb_t = 1000

    res_1 = compute_solutions_nan_filtred(param, nb_t, w_init, 1, None, addi)  # pas de modèle pour la solution fine
    print("res_1", res_1.shape, tf.reduce_mean(tf.abs(res_1)))
    print(f"{res_1.shape[1]}/{w_init.shape[0]} non-nan solution")

    if with_model:
        model_D = Model(2, True)
        model_L = Model(2, False)
    else:
        model_D = None
        model_L = None

    res_coarse_1 = compute_solutions_nan_filtred(param, nb_t, w_init_coarse, 1, model_D, addi)
    print("res_coarse_1", res_coarse_1.shape, tf.reduce_mean(tf.abs(res_coarse_1)))
    print(f"{res_coarse_1.shape[1]}/{w_init_coarse.shape[0]} non-nan solution")

    res_2 = compute_solutions_nan_filtred(param, nb_t, w_init, 2, None, addi)  # pas de modèle pour la solution fine
    print("res_2", res_2.shape, tf.reduce_mean(tf.abs(res_2)))
    print(f"{res_2.shape[1]}/{w_init.shape[0]} non-nan solution")

    res_coarse_2 = compute_solutions_nan_filtred(param, nb_t, w_init_coarse, 2, model_L, addi)
    print("res_coarse_2", res_coarse_2.shape, tf.reduce_mean(tf.abs(res_coarse_2)))
    print(f"{res_coarse_2.shape[1]}/{w_init_coarse.shape[0]} non-nan solution")

    fig, ax = plt.subplots(2, 2)
    for t in range(0, nb_t, nb_t // 10):
        color = "r" if t == 0 else "k"
        alpha = 1 if t == 0 else t / nb_t
        ax[0, 0].set_title("order 1")
        ax[0, 0].plot(res_1[t, 0, :, 0], color, alpha=alpha)

        ax[0, 1].set_title("order 1 coarse")
        ax[0, 1].plot(res_coarse_1[t, 0, :, 0], color, alpha=alpha)

        ax[1, 0].set_title("order 2")
        ax[1, 0].plot(res_2[t, 0, :, 0], color, alpha=alpha)

        ax[1, 1].set_title("order 2 coarse")
        ax[1, 1].plot(res_coarse_2[t, 0, :, 0], color, alpha=alpha)

    fig.tight_layout()
    plt.show()


def test_nan_solution():
    # pour voir des nan il faut augmenter l'échelle des conditions initiales
    param = Param(nx=1000, problem="burger")

    funcGen = FuncGenerator(param, 200, False)

    w_init = funcGen()
    w_init_coarse = Projecter(param.nx_ratio).projection_3D(w_init)
    nb_t = 1000

    ws = compute_solutions_accelerate(param, nb_t, w_init, 1, None, False)
    nan_places = tf.math.is_nan(tf.reduce_sum(ws, axis=[0, 2, 3]))

    ws_coarse = compute_solutions_accelerate(param, nb_t, w_init_coarse, 1, None, False)
    nan_places_coarse = tf.math.is_nan(tf.reduce_sum(ws_coarse, axis=[0, 2, 3]))

    nan_places_commun = tf.logical_and(nan_places_coarse, nan_places)

    def count(place):
        return tf.reduce_sum(tf.cast(place, tf.int32))

    nb_nan, nb_nan_coarse, nb_nan_commun = count(nan_places), count(nan_places_coarse), count(nan_places_commun)
    print("nb_nan,nb_nan_coarse,nb_nan_commun", nb_nan, nb_nan_coarse, nb_nan_commun)

    if nb_nan == 0:
        print("pas de nan, tant mieux")
        return

    res_where_nan = tf.boolean_mask(ws, nan_places, axis=1)
    coarse_where_nan = tf.boolean_mask(ws_coarse, nan_places, axis=1)

    fig, ax = plt.subplots(2, 2)
    for t in range(0, nb_t, nb_t // 10):
        color = "r" if t == 0 else "k"
        alpha = 1 if t == 0 else t / nb_t
        ax[0, 0].set_title("order 1")
        ax[0, 0].plot(res_where_nan[t, 0, :, 0], color, alpha=alpha)

        ax[0, 1].set_title("order 1 coarse")
        ax[0, 1].plot(coarse_where_nan[t, 0, :, 0], color, alpha=alpha)

    fig.tight_layout()
    plt.show()


#
# def test_flux():
#     from Euler.neural_network import Model
#     param = Param(nx=1000, problem="burger")
#     w_init = tf.sin(2 * np.pi * param.xs[tf.newaxis, :, tf.newaxis])
#
#     model=Model()
#     p2=model.get_shrinkage()
#     assert p2%2==0
#     Fnum=flux_with_diffu_order1(param,w_init)
#     print(Fnum.shape)


if __name__ == "__main__":
    # test_the_3_paddings()
    # test_compute_solution(True,True)
    # test_flux()
    # test_nan_solution()
    score_with_classic()
