# from Euler.core_solver import Param, GenParam
# import popup_lib.popup as pop
# import Euler.core_solver as core
# from Euler.backend import K
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# import time
# import Euler.neural_networks as nn
# from typing import *
#
# #import pandas as pd
# np.set_printoptions(precision=3, linewidth=100000)
# pp = print
#
#
# # todo: mettre le coef de pénalisation en famparam
#
# class Agent_addi_old(pop.Abstract_Agent):
#
#     def __init__(self, param: Param, model,
#                  lossCoef_stab=1,
#                  lossCoef_ridge=1e-1,
#                  lossCoef_disHLL=0.1,
#                 #  lossCoef_Laplacian=0.1,
#                 #  lossCoef_plateau=0.1,
#                  train_batch_size=256,
#                  nb_optimization=20,
#                  watch_duration=5  # durée pendant laquelle on observe la dynamique
#                  ):
#
#         self.famparams = {
#             "lossCoef_stab": lossCoef_stab,
#             "lossCoef_ridge": lossCoef_ridge,
#             # "lossCoef_Laplacian": lossCoef_Laplacian,
#             # "lossCoef_plateau": lossCoef_plateau,
#             "lossCoef_disHLL": lossCoef_disHLL,
#             "watch_duration": watch_duration
#         }
#
#         self.param = param
#         self.model: nn.Difference_model_addi = model
#         self.train_batch_size = train_batch_size
#         self.nb_optimization = nb_optimization
#
#         self.k_32 = K("tf", 32)
#         self.optimizer = tf.keras.optimizers.Adam(1e-3)
#
#     def set_and_perturb_famparams(self, famparam, period_count: int):
#         self.famparams = famparam
#         self.famparams["lossCoef_stab"] *= np.random.choice([0.7, 1.5])
#         self.famparams["lossCoef_ridge"] *= np.random.choice([0.7, 1.5])
#         self.famparams["lossCoef_disHLL"] *= np.random.choice([0.7, 1.5])
#         # self.famparams["lossCoef_Laplacian"] *= np.random.choice([0.7, 1.5])
#         # self.famparams["lossCoef_plateau"] *= np.random.choice([0.7, 1.5])
#
#         watch_duration = self.famparams["watch_duration"] + np.random.choice([-2, 2])
#         watch_duration = np.clip(watch_duration, 4, len(self.ws_val) - 1)
#         self.famparams["watch_duration"] = watch_duration
#
#     def perturb_famparams_on_decadence(self, period_count):
#         # c'est le seul coef qui fait quelque chose contre le sur-apprentissage
#         self.famparams["lossCoef_ridge"] *= 1.5
#         # on imagine qu'on est tombé dans un minimum local. On augmente les learning rate pour en sortir
#         self.famparams["lossCoef_disHLL"] *= 1.5
#         self.famparams["lossCoef_stab"] *= 1.5
#         # self.famparams["lossCoef_Laplacian"] *= 1.5
#         # self.famparams["lossCoef_plateau"] *= 1.5
#
#     def _load_ws(self, w, Y):
#         nb_t = len(w)
#         assert nb_t > self.famparams["watch_duration"] + 1, f"le nb_t:{nb_t} est trop petit par rapport à self.watch_duration: {self.famparams['watch_duration']}"
#         assert w.shape == (nb_t, self.param.batch_size, self.param.nx, 3)
#         # Y est une différence de flux, donc sa longueur en  x est  param.nx
#         assert Y.shape == (nb_t, self.param.batch_size, self.param.nx, 3), "Y.shape is:" + str(Y.shape)
#         self.k_32.check_mine(w)
#         self.k_32.check_mine(Y)
#
#         projecteur = core.Projecter(self.param.nx_ratio, 32)
#         W_proj = []
#         Y_proj = []
#         for x, y in zip(w, Y):
#             W_proj.append(projecteur.projection_3D(x))
#             Y_proj.append(projecteur.projection_3D(y))
#         W_proj = tf.stack(W_proj)
#         Y_proj = tf.stack(Y_proj)
#
#         return w, Y, W_proj, Y_proj
#
#     def load_ws_val(self, w, Y):
#         self.ws_val, self.Y_val, self.ws_val_coarse, Y_val_proj = self._load_ws(w, Y)
#
#     def load_ws_train(self, w, Y):
#         self.ws_train, self.Y_train, self.ws_train_coarse, self.Y_train_proj = self._load_ws(w, Y)
#
#     def select_time_and_batch(self, w, Y, batch_size):
#         nb_t = len(w)
#
#         # tirage des temps
#         t_init = np.random.randint(0, nb_t - self.famparams["watch_duration"] - 1, size=batch_size)
#         t_final = t_init + self.famparams["watch_duration"]
#         # tirages de batch
#         rand_b = np.random.randint(0, self.param.batch_size, batch_size)
#
#         pair_init = tf.stack([t_init, rand_b], axis=1)
#         pair_final = tf.stack([t_final, rand_b], axis=1)
#
#         w_init = tf.gather_nd(w, pair_init)
#         w_final = tf.gather_nd(w, pair_final)
#
#         y_init = tf.gather_nd(Y, pair_init)
#         y_final = tf.gather_nd(Y, pair_final)
#
#         return w_init, w_final, y_init, y_final
#
#     def HLL_error(self):
#         return self._error(False)
#
#     def agent_score(self):
#         HLL_error = self._error(False)
#         agent_error = self._error(True)
#         return HLL_error / agent_error
#
#     def _error(self, is_model, return_Ws=False):
#         if self.Y_val is None:
#             raise Exception("Vous n'avez pas charger les données de validation")
#
#         # on prend toutes les données disponibles (donc on n'utilise pas self.train_batch_size).
#         # on prend les 2 temps les plus éloignés possibles (donc on n'utilise pas self.watch_duration)
#         w_init = self.ws_val[0, :, :, :]
#         w_final = self.ws_val[-1, :, :, :]
#
#         projecteur = core.Projecter(self.param.nx_ratio, 32)
#         w_final_projected = projecteur.projection_3D(w_final)
#         w_coarse = projecteur.projection_3D(w_init)
#
#         for t in range(len(self.ws_val)):
#             var = core.Var_burger(w_coarse, self.param, self.k_32)
#             Fnum_HLL = core.Flux_HLLC(var, self.param, self.k_32)
#             dFnum_HLL = (Fnum_HLL[:, 1:, :] - Fnum_HLL[:, :-1, :])
#
#             if is_model:
#                 X = var.get_augmentation()
#                 addi = self.model(X)
#                 dFnum = dFnum_HLL + addi
#             else:
#                 dFnum = dFnum_HLL
#
#             w_coarse = w_coarse - self.param.dt_over_dx_coarse * dFnum
#
#         # error = tf.reduce_mean(tf.abs((w_coarse - w_final_projected)))
#         error = tf.reduce_mean((w_coarse - w_final_projected)**2)
#         # error = tf.reduce_max(tf.abs((w_coarse - w_final_projected)))
#
#         if return_Ws:
#             return w_coarse, w_final_projected
#         else:
#             return error.numpy()
#
#     def give_results(self, is_model):
#         w_coarse, w_final_projected = self._error(is_model, True)
#         return w_coarse, w_final_projected
#
#
#     def optimize(self):
#         # self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory, self.loss_Laplacian_forHistory = 0, 0, 0, 0
#         # self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory, self.loss_plateau_forHistory = 0, 0, 0, 0
#         self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory = 0, 0, 0
#
#         projecteur = core.Projecter(self.param.nx_ratio, 32)
#         w_init, w_final, y_init, y_final = self.select_time_and_batch(self.ws_train, self.Y_train, self.train_batch_size)
#
#         w_coarse = projecteur.projection_3D(w_init)
#
#         with tf.GradientTape() as tape:
#             dist_to_HLL = tf.constant(0.)
#             dist_plateau = tf.constant(0.)
#
#             for t in range(self.famparams["watch_duration"]):
#                 var = core.Var_burger(w_coarse, self.param, self.k_32)
#                 Fnum_HLL = core.Flux_HLLC(var, self.param, self.k_32)
#                 dFnum_HLL = (Fnum_HLL[:, 1:, :] - Fnum_HLL[:, :-1, :])
#
#                 X = var.get_augmentation()
#                 addi = self.model(X)
#                 dFnum = dFnum_HLL + addi
#
#                 if self.famparams["lossCoef_disHLL"] > 1e-6:
#                     dist_to_HLL += tf.reduce_mean(addi**2)
#
#                 # if self.famparams["lossCoef_plateau"] > 1e-6:
#                 #     dist_plateau += tf.reduce_mean((dFnum / (dFnum_HLL + 1e-8))**2)
#
#                 w_coarse = w_coarse - self.param.dt_over_dx_coarse * dFnum
#
#             """ à la première itération (t=0) w est ici égal à w[t_init+1]
#             donc à dernière itération (t=watch_duration-1), w vaut w[t_init+watch_duration]"""
#
#             loss = tf.reduce_mean((w_coarse - projecteur.projection_3D(w_final)) ** 2)
#             # loss = tf.reduce_max(tf.math.abs(w_coarse - projecteur.projection_3D(w_final)))
#             self.loss_stab_forHistory = loss.numpy()
#             loss *= self.famparams["lossCoef_stab"]
#
#             if self.famparams["lossCoef_disHLL"] > 1e-6:
#                 self.loss_disHLL_forHistory = dist_to_HLL.numpy()
#                 # pour faire une moyenne temporelle on divise par la watch duration
#                 dist_to_HLL *= self.famparams["lossCoef_disHLL"] / self.famparams["watch_duration"]
#                 loss += dist_to_HLL
#
#             if self.famparams["lossCoef_ridge"] > 1e-6:
#                 self.ridge_term = tf.constant(0.)
#                 nb_var = tf.constant(0.)
#                 for var in self.model.keras_model.trainable_variables:
#                     if len(var.shape) == 2:  # pour ne pas prendre les biais
#                         nb_var += var.shape[0] * var.shape[1]
#                         self.ridge_term += tf.reduce_sum(var ** 2)
#                 # on divise par sqrt(nombre_de_variable) pour avoir toujours le même ordre de grandeur
#                 # quelque soit la structure (théorème central limite)
#                 self.ridge_term = self.ridge_term / tf.sqrt(nb_var)
#
#                 self.loss_ridge_forHistory = self.ridge_term.numpy()
#                 self.ridge_term *= self.famparams["lossCoef_ridge"]
#                 loss += self.ridge_term
#
#             # if self.famparams["lossCoef_Laplacian"] > 1e-6:
#             #     Laplacian = (w_coarse[:, 2:, :] - 2 * w_coarse[:, 1:-1, :] + w_coarse[:, :-2, :])
#             #     Laplacian_term = tf.reduce_mean(- Laplacian * w_coarse[:, 1:-1, :]) * self.param.nx_coarse**2
#
#             #     self.loss_Laplacian_forHistory = Laplacian_term.numpy()
#             #     Laplacian_term *= self.famparams["lossCoef_Laplacian"]
#             #     # loss += Laplacian_term
#
#             # if self.famparams["lossCoef_plateau"] > 1e-6:
#             #     self.loss_plateau_forHistory = dist_plateau.numpy()
#             #     # pour faire une moyenne temporelle on divise par la watch duration
#             #     dist_plateau *= self.famparams["lossCoef_plateau"] / self.famparams["watch_duration"]
#             #     loss += dist_plateau
#
#         gradients = tape.gradient(loss, self.model.keras_model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.keras_model.trainable_variables))
#
#         # return self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory, self.loss_Laplacian_forHistory
#         # return self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory, self.loss_plateau_forHistory
#         return self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory
#
#     def to_register_at_period_end(self) -> Dict[str, float]:
#         return {
#             "loss_stab": self.loss_stab_forHistory,
#             "loss_disHLL": self.loss_disHLL_forHistory,
#             "loss_ridge": self.loss_ridge_forHistory,
#             # "loss_Laplacian": self.loss_Laplacian_forHistory
#             # "loss_plateau": self.loss_plateau_forHistory
#         }
# #
#
#     def optimize_and_return_score(self) -> float:
#         for _ in range(self.nb_optimization):
#             self.optimize()
#         return self.agent_score()
#
#     def get_famparams(self):
#         return self.famparams
#
#     def set_weights(self, weights: List):
#         self.model.keras_model.set_weights(weights)
#
#     def get_copy_of_weights(self) -> List:
#         return self.model.keras_model.get_weights()
#
#
#
#
#
# def test_loss():
#
#     param = Param(1000, 32, BC=Param.BC_periodic)
#     genParam = GenParam(param, GenParam.kind_loop)
#     nb_t = 10
#     agent = Agent_addi(param, nn.Difference_model_addi(param, 5),
#                        lossCoef_stab=0,
#                        lossCoef_disHLL=10,
#                        lossCoef_ridge=0,
#                     #    lossCoef_Laplacian=0,
#                     #    lossCoef_plateau=0,
#                        )
#     ws_train, Y_train = core.compute_fine_solutions_3(genParam, param, nb_t)
#     agent.load_ws_train(ws_train, Y_train)
#     ws_val, Y_val = core.compute_fine_solutions_3(genParam, param, nb_t)
#     agent.load_ws_val(ws_val, Y_val)
#
#     # losses_HLL, losses_stab, HLL_errors, losses_ridge, losses_Laplacian, scores_stab = [], [], [], [], [], []
#     # losses_HLL, losses_stab, HLL_errors, losses_ridge, losses_plateau, scores_stab = [], [], [], [], []
#     losses_HLL, losses_stab, HLL_errors, losses_ridge, scores_stab = [], [], [], [], []
#
#     for _ in range(15):
#         # loss_stab, loss_HLL, loss_ridge, loss_Laplacian = agent.optimize()
#         # loss_stab, loss_HLL, loss_ridge, loss_plateau = agent.optimize()
#         loss_stab, loss_HLL, loss_ridge = agent.optimize()
#
#         # if not np.isnan(loss_HLL) and not np.isnan(loss_stab) and not np.isnan(loss_Laplacian):
#         if not np.isnan(loss_HLL) and not np.isnan(loss_stab) and not np.isnan(loss_plateau):
#             # if not np.isnan(loss_HLL) and not np.isnan(loss_stab):
#             losses_HLL.append(loss_HLL)
#             losses_stab.append(loss_stab)
#             HLL_errors.append(agent.HLL_error())
#             scores_stab.append(agent.agent_score())
#             losses_ridge.append(loss_ridge)
#             # losses_Laplacian.append(loss_Laplacian)
#             # losses_plateau.append(loss_plateau)
#         else:
#             print("une des loss est nan:")
#             # print("loss_HLL, loss_stab, loss_Laplacian:", loss_HLL, loss_stab, loss_Laplacian)
#             # print("loss_HLL, loss_stab, loss_plateau:", loss_HLL, loss_stab, loss_plateau)
#             print("loss_HLL, loss_stab:", loss_HLL, loss_stab)
#
#     # fig, axs = plt.subplots(6, 1)
#     fig, axs = plt.subplots(5, 1)
#
#     nb = range(len(losses_HLL))
#
#     axs[0].plot(nb, losses_HLL, '.')
#     axs[1].plot(nb, losses_stab, '.')
#     axs[2].plot(nb, losses_ridge, '.')
#     axs[3].plot(nb, HLL_errors, '.')
#     axs[4].plot(nb, scores_stab, '.')
#     # axs[5].plot(nb, losses_Laplacian, '.')
#     # axs[5].plot(nb, losses_plateau, '.')
#
#     axs[0].set_title('losses_HLL')
#     axs[1].set_title('losses_stab')
#     axs[2].set_title('losses_ridge')
#     axs[3].set_title('HLL_errors')
#     axs[4].set_title('scores_stab')
#     # axs[5].set_title('losses_Laplacian')
#     # axs[5].set_title('losses_plateau')
#
#     plt.show()
#
#
# def test_selection():
#     data_t0 = tf.constant([[0, 1, 2], [3, 4, 5]])
#     data_t1 = data_t0 * 10
#     data_t2 = data_t0 * 100
#     X = tf.stack([data_t0, data_t1, data_t2])
#
#     t_init = [0, 0, 0]
#     t_final = [2, 2, 2]
#
#     rand_b = [0, 1, 1]
#
#     pair_init = tf.stack([t_init, rand_b], axis=1)
#     pair_final = tf.stack([t_final, rand_b], axis=1)
#
#     x_init = tf.gather_nd(X, pair_init)
#     x_final = tf.gather_nd(X, pair_final)
#
#     print("X\n", X)
#     print("pair_init\n", pair_init)
#     print("x_init\n", x_init)
#     print("x_final\n", x_final)
#
#
# if __name__ == "__main__":
#     test_loss()
