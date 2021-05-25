from Euler.core_solver import Param, GenParam
import popup_lib.popup as pop
import Euler.core_solver as core
from Euler.backend import K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import Euler.neural_networks as nn
from typing import *

#import pandas as pd
np.set_printoptions(precision=3, linewidth=100000)
pp = print


#todo: couper en plusieurs tous si trop gros
@tf.function
def _load_WY_accelerated2(W,nx_ratio,nx_coarse):
    print(f"traçage de la méthode _load_WY_accelerated2 pour les paramètres: W:{W.shape}, nx_ratio:{nx_ratio}")
    (t_max, batch_size, nx, features) = W.shape
    projecteur = core.Projecter(nx_ratio, 32)
    W = tf.reshape(W, [t_max*batch_size, nx, 3])
    W_proj = projecteur.projection(W)
    W_proj = tf.reshape(W_proj, [t_max,batch_size,nx_coarse,3])
    return W_proj


class Agent_addi(pop.Abstract_Agent):

    def __init__(self, param: Param, model,
                 lossCoef_stab=1.,
                 lossCoef_ridge=1e-1,
                 lossCoef_disHLL=0.1,
                 train_batch_size=256,
                 nb_optimization=20,
                 watch_duration=30,  # durée pendant laquelle on observe la dynamique
                 ):

        self.famparams = {
            "lossCoef_stab": float(lossCoef_stab),
            "lossCoef_ridge": float(lossCoef_ridge),
            "lossCoef_disHLL": float(lossCoef_disHLL),
            "watch_duration": int(watch_duration)
        }

        self.param = param
        self.model= model
        self.train_batch_size = train_batch_size
        self.nb_optimization = nb_optimization

        self.k_32 = K("tf", 32)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    def set_and_perturb_famparams(self, famparam, period_count: int):
        self.famparams = famparam
        self.famparams["lossCoef_stab"] *= np.random.choice([0.7, 1.5])
        self.famparams["lossCoef_ridge"] *= np.random.choice([0.7, 1.5])
        self.famparams["lossCoef_disHLL"] *= np.random.choice([0.7, 1.5])

        watch_duration = self.famparams["watch_duration"] + np.random.choice([-2, 2])
        watch_duration = np.clip(watch_duration, 4, len(self.W_val) - 1)
        self.famparams["watch_duration"] = watch_duration

    #
    # def perturb_famparams_on_decadence(self, period_count):
    #     # c'est le seul coef qui fait quelque chose contre le sur-apprentissage
    #     self.famparams["lossCoef_ridge"] *= 1.5
    #     # on imagine qu'on est tombé dans un minimum local. On augmente les learning rate pour en sortir
    #     self.famparams["lossCoef_disHLL"] *= 1.5
    #     self.famparams["lossCoef_stab"] *= 0.5

    def _load_WY(self, W):
        nb_t = len(W)
        assert nb_t > self.famparams["watch_duration"] + 1, f"le nb_t:{nb_t} est trop petit par rapport à self.watch_duration: {self.famparams['watch_duration']}"
        assert (W.shape[2],W.shape[3]) == ( self.param.nx, 3)
        #return W, self._load_WY_accelerated(W,tf.constant(len(W),tf.int32),W.shape[1])
        return W, _load_WY_accelerated2(W,self.param.nx_ratio,self.param.nx_coarse)

    def load_WY_valid(self, W):
        self.W_val, self.W_val_proj = self._load_WY(W)

    def load_WY_train(self, W):
        self.W_train,  self.W_train_proj = self._load_WY(W)

    def select_time_and_batch(self, W, train_batch_size):
        nb_t = len(W)

        # tirage des temps
        t_init = np.random.randint(0, nb_t - self.famparams["watch_duration"] - 1, size=train_batch_size)
        t_final = t_init + self.famparams["watch_duration"]
        # tirages de batch
        rand_b = np.random.randint(0, W.shape[1], train_batch_size)

        pair_init = tf.stack([t_init, rand_b], axis=1)
        pair_final = tf.stack([t_final, rand_b], axis=1)

        w_init = tf.gather_nd(W, pair_init)
        w_final = tf.gather_nd(W, pair_final)

        return w_init, w_final

    def agent_score(self):
        HLL_error_1,HLL_error_2,HLL_error_infty = self._error(False)
        agent_error_1,agent_error_2,agent_error_infty = self._error(True)
        self.score_l1=HLL_error_1/agent_error_1
        self.score_l2=HLL_error_2/agent_error_2
        self.score_linfty=HLL_error_infty/agent_error_infty

        return (self.score_l1+self.score_l2+self.score_linfty)/3

    def _error(self, is_model):
        if self.W_val_proj is None:
            raise Exception("Vous n'avez pas charger les données de validation")

        # on prend toutes les données disponibles (donc on n'utilise pas self.train_batch_size).
        # on prend les 2 temps les plus éloignés possibles (donc on n'utilise pas self.watch_duration)

        w_init = self.W_val_proj[0, :, :, :]
        duration_tensor=tf.constant(len(self.W_val),tf.int32)
        BC_solver, gamma, dt_over_dx_coarse=self.param.BC_solver, self.param.gamma, self.param.dt_over_dx_coarse

        res= self._predict_accelerated(w_init, duration_tensor, is_model, BC_solver, gamma, dt_over_dx_coarse)

        diff_abs= tf.abs(res-self.W_val_proj)

        return tf.reduce_mean(diff_abs).numpy(),tf.reduce_mean(tf.reduce_mean(diff_abs)**2).numpy(),tf.reduce_max(diff_abs).numpy()

    def predict(self):
        w_init=self.W_val_proj[0,:,:,:]
        duration_tensor=tf.constant(len(self.W_val), tf.int32)
        BC_solver, gamma, dt_over_dx_coarse=self.param.BC_solver, self.param.gamma, self.param.dt_over_dx_coarse
        res_model= self._predict_accelerated(w_init, duration_tensor, True, BC_solver, gamma, dt_over_dx_coarse)
        res_HLLC= self._predict_accelerated(w_init, duration_tensor, False, BC_solver, gamma, dt_over_dx_coarse)

        return self.W_val_proj,res_HLLC,res_model


    @tf.function
    def _predict_accelerated(self,w_init,duration_tensor, is_model,  BC_solver, gamma, dt_over_dx_coarse):
        print("traçage de la méthode _predict_accelerated avec les arguments primitifs")
        print("\t\tis_model,  BC_solver, gamma, dt_over_dx_coarse")
        print("\t\t",is_model,  BC_solver, gamma, dt_over_dx_coarse)
        print("\t\tet le tenseur w_init de shape",w_init.shape)

        res= tf.TensorArray(tf.float32, size=duration_tensor, element_shape=[w_init.shape[0], self.param.nx_coarse, 3], dynamic_size=False, clear_after_read=True)

        for t in tf.range(duration_tensor):
            res=res.write(t, w_init)

            var = core.Var(w_init, BC_solver,gamma, self.k_32)
            Fnum_HLL = core.Flux_HLLC(var, gamma, self.k_32)
            dFnum_HLL = (Fnum_HLL[:, 1:, :] - Fnum_HLL[:, :-1, :])

            if is_model:
                X = var.get_augmentation()
                addi = self.model(X)
                dFnum = dFnum_HLL + addi
            else:
                dFnum = dFnum_HLL

            w_init = w_init - dt_over_dx_coarse * dFnum

        return res.stack()

    @tf.function
    def gradient_tape(self, w_init, w_final, watch_duration_tensor, lossCoef_stab, lossCoef_disHLL, lossCoef_ridge, BC_solver, gamma, dt_over_dx_coarse):
        print(f"Traçage de la méthode gradient_tape avec les tenseurs w_init:{w_init.shape}, w_final:{w_final.shape}")
        print("\t\tet les primitifs BC_solver, gamma, dt_over_dx_coarse")
        print("\t\t",BC_solver, gamma, dt_over_dx_coarse)

        w_coarse=w_init
        with tf.GradientTape() as tape:
            dist_to_HLL = tf.constant(0.)
            for _ in tf.range(watch_duration_tensor):
                var = core.Var(w_coarse, BC_solver,gamma, self.k_32)
                Fnum_HLL = core.Flux_HLLC(var, gamma, self.k_32)
                dFnum_HLL = (Fnum_HLL[:, 1:, :] - Fnum_HLL[:, :-1, :])

                X = var.get_augmentation()
                addi = self.model(X)
                dFnum = dFnum_HLL + addi

                if lossCoef_disHLL > 1e-6:
                    dist_to_HLL += tf.reduce_mean(tf.square(addi))

                w_coarse = w_coarse - dt_over_dx_coarse * dFnum

            """ à la première itération (t=0) W est ici égal à W[t_init+1]
            donc à dernière itération (t=watch_duration-1), W vaut W[t_init+watch_duration]"""

            loss_stab_forHistory = tf.reduce_mean(tf.square(w_coarse-w_final))#tf.reduce_mean(tf.square(w_coarse - w_final))
            loss = loss_stab_forHistory*lossCoef_stab

            loss_disHLL_forHistory=tf.constant(0.)
            if lossCoef_disHLL > 1e-6:
                # pour faire une moyenne temporelle on divise par la watch duration
                loss_disHLL_forHistory= dist_to_HLL / watch_duration_tensor
                loss += loss_disHLL_forHistory*lossCoef_disHLL


            loss_ridge_forHistory=tf.constant(0.)
            ridge_term=tf.constant(0.) #nécessaire de déclarer la variable pour dresser le graph des calculs (comme en C++)
            if  lossCoef_ridge> 1e-6:
                nb_var = tf.constant(0.)
                for var in self.model.keras_model.trainable_variables:
                    if len(var.shape) == 2:  # pour ne pas prendre les biais
                        nb_var += var.shape[0] * var.shape[1]
                        ridge_term += tf.reduce_sum(tf.square(var))
                #todo: avant: on divise par sqrt(nombre_de_variable) pour avoir toujours le même ordre de grandeur
                # quelque soit la structure (théorème central limite)
                loss_ridge_forHistory = ridge_term / nb_var
                loss += loss_ridge_forHistory*lossCoef_ridge


        gradients = tape.gradient(loss, self.model.keras_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.keras_model.trainable_variables))

        return loss_stab_forHistory, loss_disHLL_forHistory, loss_ridge_forHistory


    def optimize(self):

        self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory = 0, 0, 0

        w_init, w_final = self.select_time_and_batch(self.W_train_proj,  self.train_batch_size)

        self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory = \
            self.gradient_tape(w_init, w_final,
                                  tf.constant(self.famparams["watch_duration"],tf.float32),
                                  tf.constant(self.famparams["lossCoef_stab"],tf.float32),
                                  tf.constant(self.famparams["lossCoef_disHLL"],tf.float32),
                                  tf.constant(self.famparams["lossCoef_ridge"],tf.float32),
                                  self.param.BC_solver, self.param.gamma, self.param.dt_over_dx_coarse
                                  )
        self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory=self.loss_stab_forHistory.numpy(), self.loss_disHLL_forHistory.numpy(), self.loss_ridge_forHistory.numpy()

        return self.loss_stab_forHistory, self.loss_disHLL_forHistory, self.loss_ridge_forHistory

    def to_register_at_period_end(self) -> Dict[str, float]:
        return {
            "loss_stab": self.loss_stab_forHistory,
            "loss_disHLL": self.loss_disHLL_forHistory,
            "loss_ridge": self.loss_ridge_forHistory,
            "loss_stab*coef": self.loss_stab_forHistory*self.famparams["lossCoef_stab"],
            "loss_disHLL*coef": self.loss_disHLL_forHistory*self.famparams["lossCoef_disHLL"],
            "loss_ridge*coef": self.loss_ridge_forHistory*self.famparams["lossCoef_ridge"],
            "score_l1":self.score_l1,
            "score_l2":self.score_l2,
            "score_linfty":self.score_linfty

            # "loss_Laplacian": self.loss_Laplacian_forHistory
            # "loss_plateau": self.loss_plateau_forHistory
        }

    def optimize_and_return_score(self) -> float:
        for _ in range(self.nb_optimization):
            self.optimize()
        return self.agent_score()


    def get_famparams(self):
        return self.famparams

    def set_weights(self, weights: List):
        self.model.keras_model.set_weights(weights)

    def get_copy_of_weights(self) -> List:
        return self.model.keras_model.get_weights()

from Euler.initial_conditions import init_periodic
def test_losses():
    nb_t = 60
    k_tf32 = K("tf", 32)
    param = Param(500, 32, BC_solver=Param.BC_periodic, BC_model=Param.BC_periodic)

    agent = Agent_addi(param, nn.Difference_model_tricky(param, 5),
                           lossCoef_stab=1.,
                           lossCoef_disHLL=0,
                           lossCoef_ridge=0,
                           )
    w_init=init_periodic(param,20)
    W_train = core.compute_solutions(param,nb_t,w_init,False,k_tf32)
    agent.load_WY_train(W_train)
    #pas besoin de loader des données de validations

    losses_stab = []
    losses_dist_HLL = []
    losses_ridge = []
    for _ in range(100):
        loss_stab, loss_dist_HLL, loss_ridge = agent.optimize()
        losses_stab.append(loss_stab)
        losses_dist_HLL.append(loss_dist_HLL)
        losses_ridge.append(loss_ridge)

        if np.isnan(loss_stab):
            print("la loss_stab est nan:")
        if np.isnan(loss_dist_HLL):
            print("la loss_dist_HLL est nan:")
        if np.isnan(loss_ridge):
            print("la loss_ridge est nan:")

    fig, axs = plt.subplots(3, 1, figsize=(9, 12))
    axs[0].plot(losses_stab)
    axs[0].set_title("losses_stab")
    axs[1].plot(losses_dist_HLL)
    axs[1].set_title("losses_dist_HLL")
    axs[2].plot(losses_ridge)
    axs[2].set_title("losses_ridge")

    plt.show()


def test_predict():
    k = K("tf", 32)
    param = Param(500, 32, BC_solver=Param.BC_periodic, BC_model=Param.BC_periodic)
    nb_t = 100
    agent = Agent_addi(param, nn.Difference_model_tricky(param, 5))

    w_init = init_periodic(param, 20)
    W_train = core.compute_solutions(param, nb_t, w_init, False, k)
    agent.load_WY_train(W_train)
    agent.load_WY_train(W_train)

    w_init = init_periodic(param, 20)
    W_val = core.compute_solutions(param, nb_t, w_init, False, k)
    agent.load_WY_valid(W_val)

    fine_proj, res_hll, res_model = agent.predict()

    fig, axs = plt.subplots(3, 1)
    ts = np.arange(param.nx_coarse)
    t = 50
    axs[0].plot(fine_proj[t, 0, :, 0], label='fine')
    axs[1].plot(res_hll[t, 0, :, 0], label='HLL')
    axs[2].plot(ts, res_model[t, 0, :, 0], ".", label='model')
    fig.legend()

    fine_proj, res_hll, res_model = agent.predict()

    fig, axs = plt.subplots(3, 1)
    ts = np.arange(param.nx_coarse)
    t=50
    axs[0].plot(fine_proj[t, 0, :, 0],label='fine')
    axs[1].plot(res_hll[t, 0, :, 0],label='HLL')
    axs[2].plot(ts,res_model[t, 0, :, 0],".",label='model')
    fig.legend()

    plt.show()




def test_loss_and_predict():
    k=K("tf",32)
    param = Param(500, 32, BC_solver=Param.BC_periodic,BC_model=Param.BC_periodic)
    nb_t = 100
    agent = Agent_addi(param, nn.Difference_model_tricky(param, 5))


    w_init = init_periodic(param, 20)
    W_train = core.compute_solutions(param, nb_t, w_init, False, k)
    agent.load_WY_train(W_train)
    agent.load_WY_train(W_train)

    w_init = init_periodic(param, 20)
    W_val = core.compute_solutions(param, nb_t, w_init, False, k)
    agent.load_WY_valid(W_val)



    losses_stab=[]

    for _ in range(100):
        loss_stab,_,_ = agent.optimize()
        if  not np.isnan(loss_stab):
            losses_stab.append(loss_stab)
        else:
            print("la loss est nan:")

    plt.plot(losses_stab)

    score=agent.agent_score()

    print("score:",score)

    fine_proj,res_hll,res_model=agent.predict()
    fig,axs=plt.subplots(3,3)
    axs=axs.flatten()
    ts=np.arange(param.nx_coarse)
    for i in range(9):
        axs[i].plot(fine_proj[-1,i,:,0],label="fine")
        axs[i].plot(res_hll[-1, i, :, 0],label="hll")
        axs[i].plot(ts,res_model[-1, i, :, 0],".",label="model")

    axs[0].legend()
    plt.show()



def test_Family_trainer():
    k = K("tf", 32)
    param = Param(500, 32, BC_solver=Param.BC_periodic, BC_model=Param.BC_periodic)

    fam_size = 2
    period_duration = "15 seconds"  # avant accélération: 60 secondes
    all_agents = []


    def family_full(window_size, color):
        name = "full_" + str(window_size)
        agents = []
        model_struct = (32, 64, 32)
        # model_struct = (16,32,16)
        for _ in range(fam_size):
            agent = Agent_addi(param, nn.Difference_model_tricky(param, window_size, model_struct=model_struct),
                                   watch_duration=20,
                                   lossCoef_stab=10.,
                                   lossCoef_ridge=3e-3,
                                   lossCoef_disHLL=1.,
                                   nb_optimization=15,
                                   )
            agents.append(agent)
            all_agents.append(agent)

        return pop.Family_trainer(
            agents=agents,
            nb_bestweights_averaged=3,
            nb_strong=3,
            period_duration=period_duration,
            name=name,
            color=color)


    family_trainers = [
        family_full(5, "red"),
        #  family_full(3,"blue"),
    ]


    def load_data():
        nb_t = 800
        ti = time.time()
        w_init = init_periodic(param, 20)
        W_train = core.compute_solutions(param, nb_t, w_init, False, k)

        w_init = init_periodic(param, 20)
        W_val = core.compute_solutions(param, nb_t, w_init, False, k)

        print("|génération des données: ", time.time() - ti, end="")

        ti = time.time()
        for agent in all_agents:
            agent.load_WY_train(W_train)
            agent.load_WY_valid(W_val)
        print("|load données: ", time.time() - ti, end="")

    try:
        for i in range(1):
            load_data()
            for family_trainer in family_trainers:
                family_trainer.period()
    except KeyboardInterrupt:
        for family_trainer in family_trainers:
            # pour pouvoir reprendre si on veut
            family_trainer.interupt_period()

def test_selection():
    data_t0 = tf.constant([[0, 1, 2], [3, 4, 5]])
    data_t1 = data_t0 * 10
    data_t2 = data_t0 * 100
    X = tf.stack([data_t0, data_t1, data_t2])

    t_init = [0, 0, 0]
    t_final = [2, 2, 2]

    rand_b = [0, 1, 1]

    pair_init = tf.stack([t_init, rand_b], axis=1)
    pair_final = tf.stack([t_final, rand_b], axis=1)

    x_init = tf.gather_nd(X, pair_init)
    x_final = tf.gather_nd(X, pair_final)

    print("X\n", X)
    print("pair_init\n", pair_init)
    print("x_init\n", x_init)
    print("x_final\n", x_final)


if __name__ == "__main__":
    #test_predict()
    #test_loss_and_predict()
    #test_Family_trainer()
    test_losses()
