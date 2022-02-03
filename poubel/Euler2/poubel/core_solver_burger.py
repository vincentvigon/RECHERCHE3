# # from tensorflow.python.types.core import Value
# from Euler2.param import Param, Projecter
# import numpy as np
# import matplotlib.pyplot as plt
# np.set_printoptions(precision=3, linewidth=100000)
# pp = print
# from Euler2.core_sorver_commun import  pad_w
# from Euler2.initial_conditions_burger import init_periodic,init_non_periodic
# import tensorflow as tf
#
#
#
# def SST_viscous(a, b):
#     """ order 3 limiter without smoothness detection [ Schmidtmann Seibold Torrilhon 2015 ] """
#
#     positive_a = a > 0
#     positive_b = b > 0
#     zeros = tf.zeros_like(a)
#
#     c1 = np.logical_and(positive_a, positive_b)
#     c2 = np.logical_and(np.logical_not(positive_a), positive_b)
#     c3 = np.logical_and(positive_a, np.logical_not(positive_b))
#     c4 = np.logical_and(np.logical_not(positive_a), np.logical_not(positive_b))
#
#     limiter = tf.where(c1, tf.maximum(zeros, tf.minimum(tf.minimum(2 * a, 3 / 2 * b), (a + 2 * b) / 3)), zeros)
#     limiter = tf.where(c2, tf.maximum(zeros, tf.minimum(- a, (a + 2 * b) / 3)), limiter)
#     limiter = tf.where(c3, tf.minimum(zeros, tf.maximum(- a, (a + 2 * b) / 3)), limiter)
#     limiter = tf.where(c4, tf.minimum(zeros, tf.maximum(tf.maximum(2 * a, 3 / 2 * b), (a + 2 * b) / 3)), limiter)
#
#     return limiter
#
#
# @tf.function
# def compute_solutions_order1_burger(param: Param, nb_t, dt_over_dx, dt_over_dx_coarse, nx, nx_coarse, BC_solver, W_init, is_coarse: bool):
#
#     print("traçage de la fonction 'compute_solutions_order1_burger' avec les arguments primitif:")
#     print("\t\t nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC,gamma,is_coarse=")
#     print("\t\t", nb_t, dt_over_dx, dt_over_dx_coarse, nx, nx_coarse, BC_solver, is_coarse)
#     print("\t\t et le tenseurs W_init de shape:", W_init.shape)
#     nx = nx_coarse if is_coarse else nx
#     res = tf.TensorArray(tf.float32, size=nb_t, element_shape=[W_init.shape[0],nx,1], dynamic_size=False, clear_after_read=True)
#
#     W = W_init  # inutile de faire une copie, elle est faites dans le Var_burger() au moment du padding
#
#     dt_over_dx = dt_over_dx_coarse if is_coarse else dt_over_dx
#
#     for t in tf.range(nb_t):
#         res = res.write(t, W)
#         var = Var_burger(W, BC_solver, param)
#         Fnum = Flux_HLLC(var)
#         dt_over_dx = dt_over_dx
#         dFnum = Fnum[:, 1:, :] - Fnum[:, :-1, :]
#         W = W - dt_over_dx * dFnum
#
#     return res.stack()
#
#
# def compute_solutions_burger(param: Param, nb_t, W_init, is_coarse):
#
#     dt_over_dx, dt_over_dx_coarse, nx, nx_coarse, BC_solver, gamma = param.dt_over_dx, param.dt_over_dx_coarse, param.nx, param.nx_coarse, param.BC_solver, param.gamma
#
#     if is_coarse:
#         assert W_init.shape[1] == param.nx_coarse
#         return compute_solutions_order1_burger(param, nb_t, dt_over_dx, dt_over_dx_coarse, nx, nx_coarse, BC_solver, W_init, True)
#     else:
#         assert W_init.shape[1] == param.nx
#         if param.order == 1:
#             return compute_solutions_order1_burger(param, nb_t, dt_over_dx, dt_over_dx_coarse, nx, nx_coarse, BC_solver, W_init, False)
#         else:
#             raise ValueError("ordre>1: TODO")
#         # elif param.order == 2:
#         #     return compute_solutions_order2(param, nb_t, W_init, k)
#         # elif param.order == 3:
#         #     return compute_solutions_order3(param, nb_t, W_init, k)
#         # else:
#         #     raise ValueError("order > 3 is not supported in compute_solutions")
#
#
# # pour les tests locaux uniquement
# # def generate_fine_solutions(genParam: GenParam, param: Param, nb_t,k:K):
# #     """ presque rien ici. """
# #     generator = FuncGenerator(genParam, param, k)
# #     W_init = generator.init_W()
# #     return compute_solutions(param, nb_t, W_init, False, k)
#
#
# def compare_with_projection_one_kind(is_periodic):
#
#     if is_periodic:
#         BC = Param.BC_periodic
#     else:
#         BC = Param.BC_reflexive
#
#     param = Param(600, BC_solver=BC)
#
#     if is_periodic:
#         w_init = init_periodic(param, 50)
#     else:
#         w_init = init_non_periodic(param, 50)
#     nb_t = 800
#     w_init_coarse = Projecter(param.nx_ratio).projection_3D(w_init)
#
#     res = compute_solutions_burger(param, nb_t, w_init, False)
#     res_coarse = compute_solutions_burger(param, nb_t, w_init_coarse, True)
#
#     print("(nb_t,batch,nx,1)=",res.shape)
#     print("(nb_t,batch,nx_coarse,1)=",res_coarse.shape)
#
#
#     fig, ax = plt.subplots(1, 2)
#
#     title = "periodic" if is_periodic else "non periodic"
#     fig.suptitle(title)
#
#     for t in range(0, nb_t, nb_t // 10):
#         color = "r" if t == 0 else "k"
#         alpha = 1 if t == 0 else t / nb_t
#         ax[0].set_title("rho")
#         ax[0].plot(res[t, 0, :, 0], color, alpha=alpha)
#         ax[1].plot(res_coarse[t, 0, :, 0], color, alpha=alpha)
#
#     fig.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#
#     compare_with_projection_one_kind(True)
