# # from tensorflow.python.types.core import Value
# from Euler2.param import  Projecter,Param
# import numpy as np
# np.set_printoptions(precision=3, linewidth=100000)
# pp = print
# import tensorflow as tf
# from Euler2.core_sorver_commun import  pad_w
#
#
# class Var_euler:
#     def __init__(self, W: tf.Tensor,p:int,param:Param,gamma):
#         """  toutes les variables _XXX sont de dimensions nx+2p """
#
#         """ signaux de longeur nx+2p  """
#         self.W = pad_w(W,param,p)
#
#         # des alias
#         self.rho = self.W[:, :, 0]  # density
#         self.rhoV = self.W[:, :, 1]  # moment
#         self.E = self.W[:, :, 2]  # energy
#
#         self.V = self.rhoV / self.rho
#
#         self.rhoVV = self.rhoV * self.V
#
#         self.p = (gamma - 1) * (self.E - 1 / 2 * self.rhoVV)  # pressure
#
#         p_over_rho = self.p / self.rho
#         # pour éviter des Nan produite par de valeur négatives (résiduelle ?)
#         p_over_rho = tf.where(p_over_rho < 0., 0., p_over_rho)
#         self.sound = tf.sqrt(gamma * p_over_rho)  # sound_speed
#
#         self.F = tf.stack([self.rhoV, self.rhoVV + self.p, self.V * (self.E + self.p)], axis=2)
#
#         aug = tf.stack([self.rho, self.rhoV, self.E, self.V, self.p, self.sound, self.F[:, :, 1], self.F[:, :, 2]], axis=2)
#
#
# def Flux_mean(var):
#     return (var.compute("Flux", "L") + var.compute("Flux", "R")) * 0.5
#
# def HLLC_Wave_Speeds(Density_L, Density_R, Velocity_L, Velocity_R, Pressure_L, Pressure_R, Sound_L, Sound_R, gamma:float):
#     Pressure_Mean = (Pressure_L + Pressure_R) / 2
#     Density_Mean = (Density_L + Density_R) / 2
#     Sound_Mean = (Sound_L + Sound_R) / 2
#
#     Pressure_Star = tf.maximum(tf.zeros_like(Pressure_Mean),
#                               Pressure_Mean - Density_Mean * Sound_Mean * (Velocity_R - Velocity_L) / 2)
#
#     condition_L = Pressure_Star > Pressure_L
#     q_L = tf.where(condition_L, tf.sqrt(1 + (gamma + 1) / (2 * gamma) * (Pressure_Star / Pressure_L - 1)), tf.ones_like(Pressure_Star))
#
#     condition_R = Pressure_Star > Pressure_R
#     q_R = tf.where(condition_R, tf.sqrt(1 + (gamma + 1) / (2 * gamma) * (Pressure_Star / Pressure_R - 1)), tf.ones_like(Pressure_Star))
#
#     return (Velocity_L - Sound_L * q_L,
#             Velocity_R + Sound_R * q_R)
#
#
# def Flux_HLLC(var: Var_euler, gamma:float):
#     # vrai flux
#     W_L = var.W[:,:-1,:]
#     W_R = var.get("w", "R")
#
#     F_L = var.F[:,:-1,:]
#     F_R = var.get("Flux", "R")
#
#     Density_L = var.rho[:,:-1,:]
#     Density_R = W_R[:, :, 0]
#
#     Momentum_L = W_L[:, :, 1]
#     Momentum_R = W_R[:, :, 1]
#
#     Velocity_L = var.get("V", "L")
#     Velocity_R = var.get("V", "R")
#
#     Pressure_L = var.get("p", "L")
#     Pressure_R = var.get("p", "R")
#
#     Sound_L = var.get("sound", "L")
#     Sound_R = var.get("sound", "R")
#
#     lamb_L, lamb_R = HLLC_Wave_Speeds(Density_L, Density_R, Velocity_L, Velocity_R, Pressure_L, Pressure_R, Sound_L, Sound_R, gamma)
#
#     Lambda_Star = ((Pressure_R - Pressure_L
#                   + Momentum_L * (lamb_L - Velocity_L)
#                   - Momentum_R * (lamb_R - Velocity_R))
#                   / (Density_L * (lamb_L - Velocity_L)
#                    - Density_R * (lamb_R - Velocity_R)))
#
#     Pressure_Star = Pressure_L + Density_L * (lamb_L - Velocity_L) * (Lambda_Star - Velocity_L)
#
#     Density_L_Star = Density_L * (lamb_L - Velocity_L) / (lamb_L - Lambda_Star)
#     W_L_Star_0 = Density_L_Star
#     W_L_Star_1 = Density_L_Star * Lambda_Star
#     W_L_Star_2 = Pressure_Star / (gamma - 1) + Density_L_Star * Lambda_Star**2 / 2
#     W_L_Star = tf.stack((W_L_Star_0, W_L_Star_1, W_L_Star_2), axis=2)
#
#     Density_R_Star = Density_R * (lamb_R - Velocity_R) / (lamb_R - Lambda_Star)
#     W_R_Star_0 = Density_R_Star
#     W_R_Star_1 = Density_R_Star * Lambda_Star
#     W_R_Star_2 = Pressure_Star / (gamma - 1) + Density_R_Star * Lambda_Star**2 / 2
#     W_R_Star = tf.stack((W_R_Star_0, W_R_Star_1, W_R_Star_2), axis=2)
#
#     lamb_L = lamb_L[:, :, np.newaxis]
#     Lambda_Star = Lambda_Star[:, :, np.newaxis]
#     lamb_R = lamb_R[:, :, np.newaxis]
#
#     F_L_Star = F_L + lamb_L * (W_L_Star - W_L)
#     F_R_Star = F_R + lamb_R * (W_R_Star - W_R)
#
#     zone_1 = (lamb_L >= 0)
#     #la zone_2 n'est pas utilisée
#     #zone_2 = tf.logical_and((Lambda_Star >= 0), (lamb_L < 0))
#     zone_3 = tf.logical_and((lamb_R >= 0), (Lambda_Star < 0))
#     zone_4 = (lamb_R < 0)
#
#     numerical_Flux = tf.where(zone_1, F_L, F_L_Star)
#     numerical_Flux = tf.where(zone_3, F_R_Star, numerical_Flux)
#     numerical_Flux = tf.where(zone_4, F_R, numerical_Flux)
#
#     return numerical_Flux
#
#
#
#
# def minmod(a, b):
#     """ minmod limiter """
#     c1 = tf.logical_and(a > 0, b > 0)
#     c2 = tf.logical_and(a < 0, b < 0)
#
#     limiter = tf.where(c1, tf.minimum(a, b), tf.zeros_like(a))
#     limiter = tf.where(c2, tf.maximum(a, b), limiter)
#
#     return limiter
#
#
# def SST_viscous(a, b):
#     """ order 3 limiter without smoothness detection [ Schmidtmann Seibold Torrilhon 2015 ] """
#
#     # zeros_N = tf.zeros(a.shape)
#     # limiter = np.copy(zeros_N)
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
# def compute_solutions_order1(nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC_solver,gamma, W_init,is_coarse:bool):
#     print("traçage de la fonction 'compute_solutions_order1' avec les arguments primitif:")
#     print("\t\t nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC,gamma,is_coarse=")
#     print("\t\t",nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC_solver,gamma,is_coarse)
#     print("\t\t et le tenseurs W_init de shape:",W_init.shape)
#
#     """ res=tf.zeros((nb_t,) + W_init.shape) ne fonctionne pas à cause de: EagerTensor' object does not support item assignment  """
#     nx= nx_coarse if is_coarse else nx
#     res = tf.TensorArray(tf.float32,size=nb_t,element_shape=[W_init.shape[0],nx,3],dynamic_size=False,clear_after_read=True)
#
#     W = W_init  # inutile de faire une copie, elle est faites dans le Var_burger() au moment du padding
#
#     dt_over_dx= dt_over_dx_coarse if is_coarse else dt_over_dx
#
#     for t in tf.range(nb_t):
#         res=res.write(t,W)
#         var = Var_euler(W, BC_solver, gamma)
#         Fnum = Flux_HLLC(var, gamma)
#         dt_over_dx = dt_over_dx
#         dFnum=Fnum[:, 1:, :] - Fnum[:, :-1, :]
#         W = W - dt_over_dx * dFnum
#
#     return res.stack()
#
# def compute_solutions(param:Param, nb_t, W_init, is_coarse):
#
#     dt_over_dx, dt_over_dx_coarse, nx, nx_coarse, BC_solver, gamma= param.dt_over_dx,param.dt_over_dx_coarse,param.nx,param.nx_coarse,param.BC_solver,param.gamma
#
#     if is_coarse:
#         assert W_init.shape[1] == param.nx_coarse
#         return compute_solutions_order1(nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC_solver,gamma, W_init,True)
#     else:
#         assert W_init.shape[1] == param.nx
#         if param.order == 1:
#             return compute_solutions_order1(nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC_solver,gamma, W_init,False)
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
#
# def test_flux_HLLC():
#
#     param = Param(600, BC_solver=Param.BC_neumann)
#     w_init = init_sod(param)
#
#
#     gamma=0.5
#     var = Var_euler(w_init, Param.BC_neumann, gamma)
#     Fnum = Flux_HLLC(var, gamma)
#
#     #w_init_coarse = Projecter(param.nx_ratio).projection_3D(w_init)
#
#     print("w_init.shape",w_init.shape)
#     print("Fnum.shape",Fnum.shape)
#
#
#
#
#
# def compare_with_projection_one_kind(is_periodic):
#
#     if is_periodic:
#         BC=Param.BC_periodic
#     else:
#         BC=Param.BC_neumann
#
#     param = Param(600,  BC_solver=BC)
#
#     if is_periodic:
#         w_init = init_periodic(param,50)
#     else:
#         w_init = init_sod(param)#init_non_periodic(param,50)
#     nb_t = 800
#     w_init_coarse = Projecter(param.nx_ratio).projection_3D(w_init)
#
#     res = compute_solutions(param, nb_t, w_init,  False)
#     res_coarse = compute_solutions(param, nb_t, w_init_coarse,True)
#
#     print(res.shape)
#
#     fig, ax = plt.subplots(3, 2)
#
#     title = "periodic" if is_periodic else "non periodic"
#     fig.suptitle(title)
#
#     for t in range(0, nb_t, nb_t // 10):
#         color = "r" if t == 0 else "k"
#         alpha = 1 if t == 0 else t / nb_t
#         ax[0, 0].set_title("rho")
#         ax[0, 0].plot(res[t, 0, :, 0], color, alpha=alpha)
#         ax[0, 1].plot(res_coarse[t, 0, :, 0], color, alpha=alpha)
#         ax[1, 0].set_title("rhoV")
#         ax[1, 0].plot(res[t, 0, :, 1], color, alpha=alpha)
#         ax[1, 1].plot(res_coarse[t, 0, :, 1], color, alpha=alpha)
#         ax[2, 0].set_title("E")
#         ax[2, 0].plot(res[t, 0, :, 2], color, alpha=alpha)
#         ax[2, 1].plot(res_coarse[t, 0, :, 2], color, alpha=alpha)
#
#     fig.tight_layout()
#     plt.show()
#
#
#
#
# def statistics_of_augmentation(kind):
#
#     param = Param(100)
#
#     batch_size = 20
#
#     if kind=="periodic":
#         BC = Param.BC_periodic
#         w_init = init_periodic(param, batch_size)
#     elif kind=="non-periodic":
#         BC = Param.BC_reflexive
#         w_init = init_non_periodic(param, batch_size)
#     else :
#         BC =Param.BC_neumann
#         w_init = init_random_sod(param,batch_size)
#
#     param.BC_model=BC
#     nb_t = 800
#     res = compute_solutions(param, nb_t, w_init, False)
#     res = tf.reshape(res, [nb_t * param.nx * batch_size, 3])
#     return res
#
#
# def statistics_of_augmentation_all():
#     res_pre=statistics_of_augmentation("periodic")
#     res_non=statistics_of_augmentation("non-periodic")
#     res_sod=statistics_of_augmentation("sod")
#
#     res=tf.concat([res_pre,res_non,res_sod],axis=1)
#
#     import seaborn as sns
#     print(res.shape)
#
#     sns.violinplot(data=res)
#     plt.show()
#
#
# if __name__ == '__main__':
#     test_flux_HLLC()
#
#     #activation_for_relexive()
#     #test_the_3_paddings()
#     #compare_with_projection_one_kind(True)
#
#     #statistics_of_augmentation_all()
