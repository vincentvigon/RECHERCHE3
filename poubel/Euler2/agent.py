import popup_lib.popup as pop
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from typing import *

from Euler2.param import Projecter,Param
from Euler2.core_sorver_commun import compute_solutions_accelerate, compute_solutions_nan_filtred, \
    flux_with_diffu_order1, flux_with_diffu_order2, one_time_step
from Euler2.neural_network import Model
from Euler2.initial_conditions_burger import FuncGenerator

np.set_printoptions(precision=3, linewidth=100000)
pp = print



class Agent(pop.Abstract_Agent):

    def __init__(self, param: Param, model:Model,
                 addi=True,
                 order=2,
                 lossCoef_stab=500.,
                 lossCoef_ridge=1e-1,
                 train_batch_size=256,
                 nb_optimization=60,
                 watch_duration=30  # durée pendant laquelle on observe la dynamique
                 ):

        self.famparams = {
            "lossCoef_stab": float(lossCoef_stab),
            "lossCoef_ridge": float(lossCoef_ridge),
            "watch_duration": int(watch_duration)
        }



        self.param = param
        self.model= model
        self.addi=addi
        self.order=order
        self.projecteur = Projecter(param.nx_ratio)

        assert (order==1 and model.shrinkage%2==1) or (order==2 and model.shrinkage%2==0), "model.shrinkage n'est pas adapté à  l'ordre du solver"

        self.equation_dim = 1 if param.problem=="burger" else 3

        # si on est certain de ne jamais produire de nan, on peut mettre à False (cela évite quelques calcul)
        self.check_nan=False

        self.train_batch_size = train_batch_size
        self.nb_optimization = nb_optimization
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    def set_and_perturb_famparams(self, famparam, period_count: int):
        self.famparams = famparam
        self.famparams["lossCoef_stab"] *= np.random.choice([0.7, 1.5])
        self.famparams["lossCoef_ridge"] *= np.random.choice([0.7, 1.5])

        watch_duration = self.famparams["watch_duration"] + np.random.choice([-2, 2])
        watch_duration = np.clip(watch_duration, 4, len(self.ws_val) - 1)
        self.famparams["watch_duration"] = watch_duration

    def load_ws_val(self, ws,ws_coarse):
        self.ws_val, self.ws_val_coarse = ws, ws_coarse

    def load_ws_train(self, ws,ws_coarse):
        self.ws_train, self.ws_train_coarse = ws, ws_coarse

    def select_time_and_batch(self, W, train_batch_size):
        #.shape = (temps, batch, spaciale, dim)
        # (batch, temps, dim)
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
        classic_error_1,classic_error_2,classic_error_infty = self._error(False)
        agent_error_1,agent_error_2,agent_error_infty = self._error(True)
        self.score_l1=classic_error_1/agent_error_1
        self.score_l2=classic_error_2/agent_error_2
        self.score_linfty=classic_error_infty/agent_error_infty
        return (self.score_l1+self.score_l2+self.score_linfty)/3


    def _error(self, is_model):
        res=self._predict(is_model)
        diff_abs= tf.abs(res - self.ws_val_coarse)
        e_1,e_2,e_oo=tf.reduce_mean(diff_abs).numpy(), tf.reduce_mean(tf.reduce_mean(diff_abs) ** 2).numpy(), tf.reduce_max(diff_abs).numpy()
        """pour les plots, les nan c'est embêtant cas il ne sont pas plotés. On peut alors utiliser une astuce comme ça ci-dessous"""
        # if np.isnan(e_1):
        #     e_1=10.
        # if np.isnan(e_2):
        #     e_2 = 10.
        # if np.isnan(e_oo):
        #     e_oo = 10.
        return e_1,e_2,e_oo

    def _predict(self,is_model):
        w_init = self.ws_val_coarse[0, :, :, :]
        nb_t=len(self.ws_val)
        if is_model:
            model = self.model
            order=self.order
        else:
            model=None
            order=2 #on prend le meilleurs comme référence

        return compute_solutions_accelerate(self.param, nb_t, w_init, order, model, self.addi)

    def predict(self):
        res_model= self._predict(True)
        res_classic=  self._predict(False)
        return self.ws_val_coarse, res_classic, res_model


    @tf.function
    def gradient_tape(self, w_init, w_final, watch_duration_tensor, lossCoef_stab, lossCoef_ridge):
        print(f"Traçage de la méthode gradient_tape avec les tenseurs w_init:{w_init.shape}, w_final:{w_final.shape}")

        """ici on travaille avec la résolution coarse. 
        Attention, le @tf.function va graver tous les paramètres globaux en dur (self.param, self.order)
        Seul les paramètres passé en argument peuvent être modifier (et normalement cela ne provoque pas de re-traçage)
        """
        w=w_init
        #
        # def one_step(w):
        #     if self.order == 1:
        #         Fnum = flux_with_diffu_order1(self.param, w, self.model, self.addi)
        #     else:
        #         Fnum = flux_with_diffu_order2(self.param, w, self.model, self.addi)
        #     dFnum = Fnum[:, 1:, :] - Fnum[:, :-1, :]
        #     return w - self.param.dt_over_dx_coarse * dFnum


        with tf.GradientTape() as tape:
            for _ in tf.range(watch_duration_tensor):
                # w_t1 = one_step(w)
                # w_t2 = one_step(w_t1)
                # w = (w + w_t2) / 2
                w=one_time_step(self.param,self.param.dt_over_dx_coarse,w,self.order,self.model,self.addi)

            w_w_final=w-w_final
            if self.check_nan:
                nan_examples=tf.math.is_nan(tf.reduce_sum(w_w_final,axis=[1,2]))
                nb_nan=tf.reduce_sum(tf.cast(nan_examples, tf.int32))
                if nb_nan>0:
                    tf.print(f"{nb_nan} exemple ont produit des nan dans le gradient tape")
                    w_w_final=tf.boolean_mask(w_w_final,tf.logical_not(nan_examples),axis=0)

            """ à la première itération (t=0) w est ici égal à w[t_init+1]
            donc à dernière itération (t=watch_duration-1), w vaut w[t_init+watch_duration]"""
            loss_stab_forHistory = tf.reduce_mean(w_w_final**2)
            loss = loss_stab_forHistory*lossCoef_stab

            loss_ridge_forHistory=tf.constant(0.)
            ridge_term=tf.constant(0.) #nécessaire de déclarer la variable pour dresser le graph des calculs (comme en C++)
            if  lossCoef_ridge> 1e-6:
                nb_var = tf.constant(0.)
                for var in self.model.trainable_variables:
                    if len(var.shape) != 1:  # pour ne pas prendre les biais
                        nb_var += var.shape[0] * var.shape[1]
                        ridge_term += tf.reduce_sum(tf.square(var))

                loss_ridge_forHistory = ridge_term / nb_var
                loss += loss_ridge_forHistory*lossCoef_ridge


        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss_stab_forHistory, loss_ridge_forHistory


    def optimize(self):

        self.loss_stab_forHistory, self.loss_ridge_forHistory = 0., 0.

        w_init, w_final = self.select_time_and_batch(self.ws_train_coarse, self.train_batch_size)

        self.loss_stab_forHistory, self.loss_ridge_forHistory = \
            self.gradient_tape(w_init, w_final,
                                  tf.constant(self.famparams["watch_duration"],tf.float32),
                                  tf.constant(self.famparams["lossCoef_stab"],tf.float32),
                                  tf.constant(self.famparams["lossCoef_ridge"],tf.float32)
                                  )
        self.loss_stab_forHistory, self.loss_ridge_forHistory=self.loss_stab_forHistory.numpy(), self.loss_ridge_forHistory.numpy()

        return self.loss_stab_forHistory, self.loss_ridge_forHistory

    def to_register_at_period_end(self) -> Dict[str, float]:
        return {
            "loss_stab": self.loss_stab_forHistory,
            "loss_ridge": self.loss_ridge_forHistory,
            "loss_stab*coef": self.loss_stab_forHistory*self.famparams["lossCoef_stab"],
            "loss_ridge*coef": self.loss_ridge_forHistory*self.famparams["lossCoef_ridge"],
            "score_l1":self.score_l1,
            "score_l2":self.score_l2,
            "score_linfty":self.score_linfty
        }

    def optimize_and_return_score(self) -> float:
        for _ in range(self.nb_optimization):
            self.optimize()
        return self.agent_score()


    def get_famparams(self):
        return self.famparams

    def set_weights(self, weights: List):
        self.model.set_weights(weights)

    def get_copy_of_weights(self) -> List:
        return self.model.get_weights()

#
# def test_predict():
#     param=Param(nx=1000,problem="burger")
#     nb_t = 100
#     model_D = Model(input_dim=2,odd_shrinkage=True)
#     agent= Agent(param, model_D, addi=True, order=1)
#
#     funcGen=FuncGenerator(param,200,False)
#
#     w_init = funcGen
#     ws=compute_solutions_nan_filtred(param,nb_t,w_init,1,None,False)
#     agent.load_ws_train(ws)
#
#     w_init = funcGen
#     ws = compute_solutions_nan_filtred(param, nb_t, w_init, 1, None, False)
#     agent.load_ws_val(ws)
#
#     fine_proj, res_classic, res_model = agent.predict()
#
#     fig, axs = plt.subplots(3, 1)
#     ts = np.arange(param.nx_coarse)
#     t = 50
#     axs[0].plot(fine_proj[t, 0, :, 0], label='fine')
#     axs[1].plot(res_classic[t, 0, :, 0], label='classic')
#     axs[2].plot(ts, res_model[t, 0, :, 0], label='model')
#     fig.legend()
#
#     plt.show()


def test_optimize_and_predict():
    param = Param(700,10)
    projecter=Projecter(param.nx_ratio)
    nb_t = 800
    model_D = Model(input_dim=2, odd_shrinkage=True)
    agent = Agent(param, model_D, addi=True, order=1)

    funcGen=FuncGenerator(param,200,False)

    w_init = funcGen()
    ws_train = compute_solutions_nan_filtred(param, nb_t, w_init, 1, None, False)
    ws_train_coarse = projecter.projection_4D(ws_train)
    w_init = funcGen()
    ws_val = compute_solutions_nan_filtred(param, nb_t, w_init, 1, None, False)
    ws_val_coarse = projecter.projection_4D(ws_val)

    print(f"{ws_train.shape[1]}/{w_init.shape[0]} examples train are non nan")
    agent.load_ws_train(ws_train,ws_train_coarse)

    print(f"{ws_val.shape[1]}/{w_init.shape[0]} examples val are non nan")
    agent.load_ws_val(ws_val,ws_val_coarse)

    losses_stab=[]
    losses_ridge=[]

    ti0=time.time()
    for _ in range(20):
        loss_stab,loss_ridge = agent.optimize()
        if  not np.isnan(loss_stab) and not np.isnan(loss_ridge):
            losses_stab.append(loss_stab)
            losses_ridge.append(loss_ridge)
        else:
            print("une des loss est nan:")

    print("durée de l'optimisation:",time.time()-ti0)

    fig,axs=plt.subplots(2,1)
    axs[0].plot(losses_stab,label="loss stab")
    axs[1].plot(losses_ridge,label="loss ridge")
    fig.legend()

    ti0=time.time()
    score=agent.agent_score()
    print("agent.score()",score,"durée du calcul:",time.time()-ti0)

    ti0=time.time()
    fine_proj, res_classic, res_model = agent.predict()
    print("durée de la prédiction:",time.time()-ti0)

    fig, ax = plt.subplots(3, 4)
    for t in range(0, nb_t, nb_t // 10):
        color = "r" if t == 0 else "k"
        alpha = 1 if t == 0 else t / nb_t
        for j in range(4):
            ax[0,j].set_title("fine proj")
            ax[0,j].plot(fine_proj[t, j, :, 0], color, alpha=alpha)

            ax[1,j].set_title("classic")
            ax[1,j].plot(res_classic[t, j, :, 0], color, alpha=alpha)

            ax[2,j].set_title("model")
            ax[2,j].plot(res_model[t, j, :, 0], color, alpha=alpha)

    fig.tight_layout()
    plt.show()



def test_Family_trainer():
    param = Param(700, 10)
    projecter=Projecter(param.nx_ratio)

    fam_size = 2
    period_duration = "2 steps"  # avant accélération: 60 secondes
    all_agents = []

    def one_family():
        agents = []
        for _ in range(fam_size):
            model_D = Model(input_dim=2,  odd_shrinkage=True)
            agent = Agent(param, model_D, addi=False, order=1)
            agents.append(agent)
            all_agents.append(agent)

        return pop.Family_trainer(
            agents=agents,
            nb_bestweights_averaged=3,
            nb_strong=3,
            period_duration=period_duration,
            )

    family_trainers = [one_family()]
    nb_examples = 200
    nb_t = 800
    funcGen=FuncGenerator(param,nb_examples,False)


    def load_data():
        w_init = funcGen()
        ws_train = compute_solutions_nan_filtred(param, nb_t, w_init, 1, None, False)
        ws_train_coarse=projecter.projection_4D(ws_train)
        w_init = funcGen()
        ws_val = compute_solutions_nan_filtred(param, nb_t, w_init, 1, None, False)
        ws_val_coarse =projecter.projection_4D(ws_val)

        nb_nan_allowed=3
        if nb_examples-ws_train.shape[0]>nb_nan_allowed or nb_examples-ws_val.shape[0]>nb_nan_allowed:
            print("ATTENTION TROP DE NAN, cela peut provoquer un grand nombre de retraçage de la fonction de projection")

        for agent in all_agents:
            agent.load_ws_train(ws_train,ws_train_coarse)
            agent.load_ws_val(ws_val,ws_val_coarse)

    try:
        for i in range(2):
            load_data()
            for family_trainer in family_trainers:
                family_trainer.period()
    except KeyboardInterrupt:
        for family_trainer in family_trainers:
            # pour pouvoir reprendre si on veut
            family_trainer.interupt_period()


    agent=family_trainers[0].get_best_agent()
    fine_proj, res_classic, res_model = agent.predict()

    fig, ax = plt.subplots(3, 4)
    for t in range(0, nb_t, nb_t // 10):
        color = "r" if t == 0 else "k"
        alpha = 1 if t == 0 else t / nb_t
        for j in range(4):
            ax[0, j].set_title("fine proj")
            ax[0, j].plot(fine_proj[t, j, :, 0], color, alpha=alpha)

            ax[1, j].set_title("classic")
            ax[1, j].plot(res_classic[t, j, :, 0], color, alpha=alpha)

            ax[2, j].set_title("model")
            ax[2, j].plot(res_model[t, j, :, 0], color, alpha=alpha)

    fig.tight_layout()
    plt.show()





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
    test_optimize_and_predict()
    #test_Family_trainer()
