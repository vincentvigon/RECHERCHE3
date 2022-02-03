import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import time
from Hamilton.separable.data_gen import Param,solver_symplectic_separable
import Hamilton.separable.preprocessing as prepro
import pandas as pd
pp=print


def make_mlp(input_dim,output_dim, struct=(50,200,50)):
    initializer = tf.keras.initializers.Orthogonal()
    inputs=tf.keras.layers.Input(shape=[input_dim])
    current=inputs
    for unit in struct:
        current=tf.keras.layers.Dense(unit,activation='tanh',kernel_initializer=initializer)(current)
    outputs = tf.keras.layers.Dense(output_dim, use_bias=False,kernel_initializer=initializer)(current)

    model= tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile() #pour éviter un warning
    return model



class Agent:

    def __init__(self,param:Param,
                 learn_directly_dH = True,
                 use_a_second_model = True,
                 watch_duration = 1,
                 preprocess_fn=None,
                 nb_times_for_val_and_test=None
                 ):
        self.param = param

        print(f"\nAgent with: learn_directly_dH:{learn_directly_dH}, use_a_second_model:{use_a_second_model},watch_duration:{watch_duration},preprocess_fn:{preprocess_fn}")

        self.learn_directly_dH = learn_directly_dH  # avec False, c'est plus lent
        self.use_a_second_model = use_a_second_model
        self.watch_duration = watch_duration  # mettre à None pour utiliser la technique classique
        self.preprocess_fn=preprocess_fn
        self.nb_times_for_val_and_test = nb_times_for_val_and_test

        self.show_plots_after_train=False

        if self.nb_times_for_val_and_test is None:
            self.nb_times_for_val_and_test=self.param.nb_t


        self.nb_times_select=10

        if self.preprocess_fn is not None:
            #ex: self.preprocess_fn=lambda a:reduce_power(a,self.compress_dim)
            q,p=self.param.initial_part_distri_val()
            q_changed=self.preprocess_fn(q)
            assert len(q_changed.shape)==2, "the result of a preprocessing must have shape=(batch_size,dim_of_preprocessing)"
            self.model_input_dim=q_changed.shape[1]
        else:
            self.preprocess_fn=lambda a:a
            self.model_input_dim=self.param.nb_particle

        #options
        """
        Ne fonctionne pas avec la config mean_recall
            self.use_a_second_model=False
        ainsi que
            self.learn_directly_dH=False  
        
        Avec la config independant
        
        Ce qui fonctionne très vite et très bien:
            self.learn_directly_dH=True
            self.use_a_second_model=False
            self.watch_duration=1 
       
        Si on met: 
            self.learn_directly_dH=False 
        Alors la seule combinaison qui fonctionne bien c'est 
            self.watch_duration=None
            self.use_a_second_model=False
        """


        if self.learn_directly_dH:
            #c'est une base-line, mais on n'est pas forcément dans le cas simplectique:
            #car une fonction générique de R^n dans R^n n'est pas forcément le gradient d'une fonction de R^n dans R
            self.model_1:tf.keras.Model = make_mlp(self.model_input_dim, self.param.nb_particle)
            self.dH1 = lambda q: self.model_1(self.preprocess_fn(q))

            if self.use_a_second_model:
                self.model_2: tf.keras.Model = make_mlp(self.model_input_dim, self.param.nb_particle)
                self.dH2 = lambda p: self.model_2(self.preprocess_fn(p))
            else:
                self.dH2 = lambda p:  p

        else:
            self.model_1: tf.keras.Model = make_mlp(self.model_input_dim, 1)
            self.dH1 = lambda q: self._der(self.model_1,q)
            #self.dH1= lambda q: self._der(self.model_1,q)

            if self.use_a_second_model:
                self.model_2: tf.keras.Model = make_mlp(self.model_input_dim, 1)
                self.dH2 = lambda p: self._der(self.model_2, p)
            else:
                self.dH2 = lambda p:  p


        self.optimizer1=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)#tf.keras.optimizers.Adam()
        self.optimizer2=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

        self.train_losses = []
        self.val_losses = []
        self.val_steps=[]

        self.best_val_loss = None
        self.best_step = None


        #validation data

        q0, p0 = self.param.initial_part_distri_val()
        self.q_val, self.p_val = solver_symplectic_separable(self.nb_times_for_val_and_test, self.param.dt, q0, p0, self.param.dH1, self.param.dH2)


    def val_step(self):
        q0, p0 = self.param.initial_part_distri_val()
        q, p = solver_symplectic_separable(self.nb_times_for_val_and_test, self.param.dt, q0, p0, self.dH1, self.dH2)
        return tf.reduce_mean((q-self.q_val)**2+(p-self.p_val)**2)


    @tf.function
    def _der(self,H,q):
        #print("traçage de la fonction de dérivation")
        with tf.GradientTape() as tape:
            tape.watch(q)
            H_q=H(self.preprocess_fn(q))
        return tape.gradient(H_q,q)


    def load_new_traj_for_train(self):
        q0, p0 = self.param.initial_part_distri_train()
        self.q_train,self.p_train=solver_symplectic_separable(self.param.nb_t,self.param.dt,q0,p0,self.param.dH1,self.param.dH2)


    def select_times(self):

        dq= (self.q_train[1:] - self.q_train[:-1]) / self.param.dt
        dp= (self.p_train[1:] - self.p_train[:-1]) / self.param.dt

        """ Attention si on respecte le schema symplectique:
                    (q^n+1-q^n)/dt= dH2(p^n)
                    (p^n+1-p^n)/dt= dH1(q^n+1)
        du coup c'est le q décallé qu'il faut considérer pour être précis. 
        """
        q = self.q_train[1:]
        p = self.p_train[:-1]

        t_select=np.random.randint(0, q.shape[0], size=self.nb_times_select)
        q_selected=tf.gather(q,t_select)
        p_selected=tf.gather(p,t_select)

        dq_selected = tf.gather(dq, t_select)
        dp_selected = tf.gather(dp, t_select)

        q_selected=tf.reshape(q_selected,[-1,self.param.nb_particle])
        p_selected = tf.reshape(p_selected, [-1,self.param.nb_particle])
        dq_selected = tf.reshape(dq_selected, [-1,self.param.nb_particle])
        dp_selected = tf.reshape(dp_selected, [-1,self.param.nb_particle])

        return q_selected,p_selected,dq_selected,dp_selected

    def select_time_intervals(self):

        t_select = np.random.randint(0, self.param.nb_t - self.watch_duration - 1, size=self.nb_times_select)

        q_init=tf.gather(self.q_train, t_select)
        p_init=tf.gather(self.p_train, t_select)

        q_final = tf.gather(self.q_train, t_select + self.watch_duration)
        p_final = tf.gather(self.p_train, t_select + self.watch_duration)

        q_init=tf.reshape(q_init,[-1,self.param.nb_particle])
        p_init=tf.reshape(p_init,[-1,self.param.nb_particle])
        q_final=tf.reshape(q_final,[-1,self.param.nb_particle])
        p_final=tf.reshape(p_final,[-1,self.param.nb_particle])

        return q_init,p_init,q_final,p_final


    def train_step(self):
        if self.watch_duration is not None:
            q_init, p_init, q_final, p_final = self.select_time_intervals()
            return self.train_step_symplectic(self.param.dt,q_init, p_init, q_final, p_final,self.watch_duration)
        else:
            q, p, dq, dp = self.select_times()
            return self.train_step_normal(q, p, dq, dp)


    @tf.function
    def train_step_normal(self,q, p, dq, dp):

        with tf.GradientTape(persistent=True) as tape:
            dq_pred =   self.dH2(p)
            dp_pred = - self.dH1(q)
            loss=tf.reduce_mean((dq-dq_pred)**2 + (dp-dp_pred)**2 )

        variables1=self.model_1.trainable_variables
        grad1=tape.gradient(loss,variables1)
        self.optimizer1.apply_gradients( zip(grad1,variables1))

        if self.use_a_second_model:
            variables2 = self.model_2.trainable_variables
            grad2=tape.gradient(loss,variables2)
            self.optimizer2.apply_gradients( zip(grad2,variables2))

        return loss


    @tf.function
    def train_step_symplectic(self,dt,q_0,p_0,q_final,p_final,watch_duration):

        q_n=q_0
        p_n=p_0
        p_n_1=p_0 #initialisation pour faire plaisir à @tf.function
        q_n_1=q_0 #idem

        with tf.GradientTape(persistent=True) as tape:

            for _ in tf.range(watch_duration):
                q_n_1= q_n + dt * self.dH2(p_n)
                p_n_1= p_n - dt * self.dH1(q_n_1)
                q_n=q_n_1
                p_n=p_n_1

            loss=tf.reduce_mean((q_n_1-q_final)**2 + (p_n_1-p_final)**2 )


        variables1=self.model_1.trainable_variables
        grad1=tape.gradient(loss,variables1)
        self.optimizer1.apply_gradients( zip(grad1,variables1))

        if self.use_a_second_model:
            variables2 = self.model_2.trainable_variables
            grad2=tape.gradient(loss,variables2)
            self.optimizer2.apply_gradients( zip(grad2,variables2))

        return loss



    def train(self,minutes=0.5):
        self.load_new_traj_for_train()
        ti0=time.time()
        step=-1
        OK=True
        while OK:
            step+=1

            train_loss=self.train_step().numpy()
            self.train_losses.append(train_loss)

            if step>0 and step%20==0:
                self.load_new_traj_for_train()

                val_loss = self.val_step().numpy()
                self.val_losses.append(val_loss)
                self.val_steps.append(step)

                if val_loss <= np.min(self.val_losses):
                    print(f"⤥:{val_loss*100:.4f}",end="")
                    self.best_val_loss = val_loss
                    self.best_step = step
                    self.model_1.save("model_1.h5") #attention, si modèle complexe, il faut indiquer un répertoire
                    if self.use_a_second_model:
                        self.model_2.save("model_2.h5")
                else:
                    print(".",end="")

                OK = (time.time() - ti0)  < minutes* 60



        self.model_1 = tf.keras.models.load_model("model_1.h5")
        if self.use_a_second_model:
            self.model_2 = tf.keras.models.load_model("model_2.h5")


        q0, p0 = self.param.initial_part_distri_test()
        q_pred, p_pred = solver_symplectic_separable(self.nb_times_for_val_and_test, self.param.dt, q0, p0, self.dH1,
                                                     self.dH2)
        q_true, p_true = solver_symplectic_separable(self.nb_times_for_val_and_test, self.param.dt, q0, p0,
                                                     self.param.dH1, self.param.dH2)
        error = tf.reduce_mean((q_pred - q_true) ** 2 + (p_pred - p_true) ** 2)

        if self.show_plots_after_train:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 4))
            ax1.scatter(self.best_step, self.best_val_loss)
            ax0.plot(np.arange(len(self.train_losses)), self.train_losses, label="train")
            ax1.plot(self.val_steps, self.val_losses, label="val")
            ax1.set_yscale('log')
            ax0.set_yscale('log')
            ax1.legend()
            ax0.legend()

            fig, (ax0,ax1) = plt.subplots(2,1,figsize=(5,10))

            print(f"test error: {error*100}")

            for i in range(self.param.nb_particle):
                ax0.plot(q_pred[:,0, i], p_pred[:,0, i])
                ax0.set_title("pred")
                ax1.plot(q_true[:, 0, i], p_true[:, 0, i])
                ax1.set_title("true")

            plt.show()

        return error.numpy()*100,(q_pred,p_pred),(q_true,p_true)






def test_models():
    mlp=make_mlp(2,2)
    X=tf.ones([5,2])
    Y=mlp.model(X)

    print(Y.shape)


def test_model_agents():
    param=Param()
    agent = Agent(param)
    X = tf.ones([5, param.nb_particle])
    Y1=agent.dH1(X)
    Y2 = agent.dH1(X)
    print(Y1.shape,Y2.shape)

    agent.train(10)


def test_agent():
    param=Param(Param.config_independant)
    param.nb_particle=12
    agent=Agent(param, use_a_second_model=True) #preprocess_fn=prepro.reduce_power2_cross2
    error=agent.train(0.5)
    print("error:",error)


def test_agent_options():

    param=Param()
    lines=[]
    for use_a_second_model in [True,False]:
        for learn_directly_dH in [True,False]:
            line={}
            lines.append(line)
            agent=Agent(param,use_a_second_model=use_a_second_model,learn_directly_dH=learn_directly_dH)
            error=agent.train(0.1)
            line["use_a_second_model"]=use_a_second_model
            line["learn_directly_dH"]=learn_directly_dH
            line["error"]=error

    compar = pd.DataFrame(lines)
    compar=compar.sort_values(by=['error'])

    print(compar)


def test_compa():
    series = dict()
    series["use_a_second_model"]=True
    series["learn_directly_dH"]=False
    series["error"]=0.9

    series2 = dict()
    series2["error"] = 0.7

    series2["use_a_second_model"] = True
    series2["learn_directly_dH"] = False

    df=pd.DataFrame([series,series2])
    df=df.sort_values(by=['error'])

    print(df)


if __name__ == "__main__":
    #test_models()
    #test_agent()
    #test_model_agents()
    #test_compa()
    test_agent_options()