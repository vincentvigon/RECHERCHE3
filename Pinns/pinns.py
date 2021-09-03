import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import clear_output
import time

np.set_printoptions(precision=2,linewidth=10000,suppress=True)


class Grid:

    def __init__(self, nb_t, max_t, nb_x, max_x):
        self.nb_t = nb_t
        self.max_t = max_t
        self.nb_x = nb_x
        self.max_x = max_x

        # un zero en float32
        zero = tf.constant(0.)
        # pour produire un linspace en float32
        self.t = tf.linspace(zero, max_t, nb_t)
        self.x = tf.linspace(zero, max_x, nb_x)

        # vecteur répétés
        x_, t_ = tf.meshgrid(self.x, self.t)
        self.t_ = tf.reshape(t_, [-1])
        self.x_ = tf.reshape(x_, [-1])

    def to_mat(self, vect):
        return tf.reshape(vect, [self.nb_t, self.nb_x])

    def plot_2d(self, U):
        U = self.to_mat(U)

        plt.imshow(U, origin="lower", cmap="jet", extent=[0, self.max_x, 0, self.max_t])
        plt.xlabel("x")
        plt.ylabel("y")

        plt.colorbar()

    def plot_several_times(self, U,initial=None):
        fig, ax = plt.subplots(figsize=(10, 3))
        nb = 20

        if initial is not None:
            ax.plot(self.x,initial,"k+",label="initial")

        for k in range(nb):
            i = int(k * self.nb_t / nb)
            ax.plot(self.x, U[i, :], color=cm.jet(k / nb), label=np.round(self.t[i].numpy(), 2))
        fig.legend()
        ax.set_xlabel("x")

    def plot_several_spaces(self, U):
        fig, ax = plt.subplots(figsize=(10, 3))
        nb = 20

        for k in range(nb):
            i = int(k * self.nb_x / nb)
            ax.plot(self.t, U[:, i], color=cm.jet(k / nb), label=np.round(self.x[i].numpy(), 2))
        fig.legend()
        ax.set_xlabel("t")

#
# class Derivator:
#
#     def __init__(self, u):
#         self.u = u
#
#     def u_t(self, t, x):
#         with tf.GradientTape() as tape:
#             tape.watch(t)
#             U = self.u(t, x)
#         U_t = tape.gradient(U, t)
#         return U_t
#
#     def u_x(self, t, x):
#         with tf.GradientTape() as tape:
#             tape.watch(x)
#             U = self.u(t, x)
#         U_x = tape.gradient(U, x)
#         return U_x
#
#     def u_tt(self, t, x):
#         with tf.GradientTape() as tape:
#             tape.watch(t)
#             U_t = self.u_t(t, x)
#         U_tt = tape.gradient(U_t, t)
#         return U_tt
#
#     def u_xx(self, t, x):
#         with tf.GradientTape() as tape:
#             tape.watch(x)
#             U_x = self.u_x(t, x)
#         U_xx = tape.gradient(U_x, x)
#         return U_xx


class Equation:

    #On triche un peu: on prend des conditions initiales qui sont des séries de Fourier
    #dont les coefficients sont connus (cela nous évitera d'avoir à les calculer)

    def __init__(self,L):
        self.L=L
        self.c=1
        self.C = [0, 2, 0, 1]
        self.B = [0, 1, 1, 1]


    def initial(self, x):

        res=tf.zeros_like(x)
        for k,Ck in enumerate(self.C):
            res+=Ck*tf.sin(k*np.pi*x/self.L)
        return res

    def initial_prime(self, x):

        res=tf.zeros_like(x)
        for k,Bk in enumerate(self.B):
            res+=k*Bk*tf.sin(k*np.pi/self.L*x)
        res*=self.c*np.pi/self.L
        return res


    def u(self, t, x):
        res=tf.zeros_like(t)
        for k,Bk in enumerate(self.B):
            res+=Bk*tf.sin(k*np.pi/self.L*x)*tf.sin(k*self.c*np.pi/self.L*t)
        for k,Ck in enumerate(self.C):
            res+=Ck*tf.sin(k*np.pi/self.L*x)*tf.cos(k*self.c*np.pi/self.L*t)
        return res


def observation():
    # une grille fine:
    L = 2
    max_t = 0.1
    g = Grid(100, max_t, 100, L)

    equation=Equation(L)

    U = equation.u(g.t_, g.x_)
    U = g.to_mat(U)


    g.plot_several_times(U,equation.initial(g.x))
    plt.show()

    g.plot_2d(U)
    plt.show()


def test_checking():
    L = 2
    max_t = 0.1
    g = Grid(100, max_t, 100, L)
    equation = Equation(L)


    t_=tf.Variable(g.t_)
    x_=tf.Variable(g.x_)

    with tf.GradientTape(persistent=True) as tape2:

        with tf.GradientTape() as tape:
            U = equation.u(t_, x_)
        U_t, U_x = tape.gradient(U, (t_, x_))

    U_tt = tape2.gradient(U_t, t_)
    U_xx = tape2.gradient(U_x, x_)

    # u(x,0)=g(x) ?
    U = g.to_mat(U)

    plt.plot(g.x, U[0, :])
    plt.plot(g.x, equation.initial(g.x), '+')

    print("almost zero:",tf.reduce_sum(U_tt-1/equation.c**2*U_xx))

    plt.show()


def make_model(input_dim):

    inputs=tf.keras.layers.Input(shape=[input_dim])
    X=inputs

    struct=(15,20,20,40,80,40,20,20,15)
    for i in struct:
        X=tf.keras.layers.Dense(i,activation="tanh")(X)
        #X=tf.keras.layers.BatchNormalization()(X)

    ouputs=tf.keras.layers.Dense(1)(X)
    return tf.keras.Model(inputs=inputs,outputs=ouputs)


#
# class Agent:
#
#     def __init__(self, L, max_t, equation:Equation):
#
#         self.L = L  # longueur de la corde
#         self.max_t = max_t  # temps maximum
#         self.equation=equation
#         self.batch_size=256
#
#         self.zero = tf.zeros([self.batch_size])
#         self.L_t = tf.ones([self.batch_size]) * self.L
#
#         self.model = make_model(2)
#
#         self.optimizer = tf.keras.optimizers.Adam(1e-3)
#
#         self.losses = []
#         self.best_loss = None
#         self.step_where_best_loss = None
#
#         self.random_sampling=True
#
#
#
#     def edp_loss(self):
#
#         batch_size=self.batch_size**2
#         if self.random_sampling:
#             t=tf.random.uniform([batch_size],tf.constant(0.),self.max_t)
#             x=tf.random.uniform([batch_size],tf.constant(0.),self.L)
#         else:
#             t = tf.linspace(tf.constant(0.), self.max_t, batch_size)
#             x = tf.linspace(tf.constant(0.), self.L, batch_size)
#
#         t = tf.Variable(t)
#         x = tf.Variable(x)
#
#         with tf.GradientTape(persistent=True) as tape2:
#
#             with tf.GradientTape() as tape:
#                 tape.watch(t)
#                 tape.watch(x)
#                 U=self.model(tf.stack([t,x],axis=1))
#             U_t,U_x = tape.gradient(U,(t,x))
#
#         U_tt=tape2.gradient(U_t,t)
#         U_xx=tape2.gradient(U_x,x)
#
#         del tape2
#
#         c2=self.equation.c**2
#         return U_tt - U_xx/c2
#
#
#
#
#     def boundary_condition_loss(self):
#
#         batch_size=self.batch_size
#         if self.random_sampling:
#             t=tf.random.uniform([batch_size],tf.constant(0.),self.max_t)
#         else:
#             t = tf.linspace(tf.constant(0.), self.max_t, batch_size)
#
#         zeros = tf.zeros_like(t)
#         Ls = tf.ones_like(t) * self.L
#
#         U0 = self.model(tf.stack([t,zeros], axis=1))
#         UL = self.model(tf.stack([t,Ls], axis=1))
#
#         return tf.reduce_sum(U0 ** 2 + UL ** 2)
#
#
#     def initial_condition_loss(self):
#
#         batch_size=self.batch_size
#         if self.random_sampling:
#             x=tf.random.uniform([batch_size],tf.constant(0.),self.L)
#         else:
#             x = tf.linspace(tf.constant(0.), self.L, batch_size)
#
#         t =  tf.zeros_like(x)
#
#         with tf.GradientTape() as tape:
#             tape.watch(t)
#             tx = tf.stack([t, x], axis=1)
#             U=self.model(tx)[:,0]
#
#         U_t=tape.gradient(U,t)
#
#         target = self.equation.initial(x)
#         target_prime = self.equation.initial_prime(x)
#
#         return tf.reduce_sum((U-target)**2)+tf.reduce_sum((U_t-target_prime)**2)
#
#
#     @tf.function
#     def step(self):
#
#         with tf.GradientTape() as tape:
#             loss = self.boundary_condition_loss()+self.initial_condition_loss()
#
#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
#
#         return loss
#
#     def optimize(self, nb_step):
#
#         for _ in range(nb_step):
#             loss = self.step().numpy()
#             self.losses.append(loss)
#
#             if self.best_loss is None or loss < self.best_loss:
#                 self.best_loss = loss
#                 self.step_where_best_loss = len(self.losses)
#             #
#             # if len(self.losses) % 500 == 0:
#             #     #clear_output(wait=True)
#             #     plt.plot(self.losses)
#             #     plt.show()
#
#         plt.plot(self.losses)
#         plt.show()

class Agent:

    def __init__(self, L, equation: Equation):

        self.L = L  # longueur de la corde
        self.equation = equation
        self.batch_size = 256

        self.zero = tf.zeros([self.batch_size])
        self.L_t = tf.ones([self.batch_size]) * self.L

        self.model = make_model(2)

        self.optimizer = tf.keras.optimizers.Adam(1e-3)

        self.losses=[]
        self.min_losses_by_period=[]
        self.min_losses_place_by_period=[]

        self.random_sampling = True

        self.delta_time=0.1
        self.current_start=0.
        self.current_end=self.current_start+self.delta_time

        #self.current_period=0
        self.step_count=-1
        self.patience=1000

    def edp_loss(self):

        batch_size = self.batch_size * 2
        if self.random_sampling:
            t = tf.random.uniform([batch_size], tf.constant(self.current_start), self.current_end)
            x = tf.random.uniform([batch_size], tf.constant(0.), self.L)
        else:
            t = tf.linspace(tf.constant(self.current_start), self.current_end, batch_size)
            x = tf.linspace(tf.constant(0.), self.L, batch_size)

        t = tf.Variable(t)
        x = tf.Variable(x)

        with tf.GradientTape(persistent=True) as tape2:

            with tf.GradientTape() as tape:
                tape.watch(t)
                tape.watch(x)
                U = self.model(tf.stack([t, x], axis=1))
            U_t, U_x = tape.gradient(U, (t, x))

        U_tt = tape2.gradient(U_t, t)
        U_xx = tape2.gradient(U_x, x)

        del tape2

        c2 = self.equation.c ** 2
        return tf.reduce_sum(U_tt - U_xx / c2)


    def boundary_condition_loss(self):

        batch_size = self.batch_size
        if self.random_sampling:
            t = tf.random.uniform([batch_size], tf.constant(self.current_start), self.current_end)
        else:
            t = tf.linspace(tf.constant(self.current_start), self.current_end, batch_size)

        zeros = tf.zeros_like(t)
        Ls = tf.ones_like(t) * self.L

        U0 = self.model(tf.stack([t, zeros], axis=1))
        UL = self.model(tf.stack([t, Ls], axis=1))

        return tf.reduce_sum(U0 ** 2 + UL ** 2)

    def initial_condition_loss(self):

        batch_size = self.batch_size
        if self.random_sampling:
            x = tf.random.uniform([batch_size], tf.constant(0.), self.L)
        else:
            x = tf.linspace(tf.constant(0.), self.L, batch_size)

        t = tf.zeros_like(x)

        with tf.GradientTape() as tape:
            tape.watch(t)
            tx = tf.stack([t, x], axis=1)
            U = self.model(tx)[:, 0]

        U_t = tape.gradient(U, t)

        target = self.equation.initial(x)
        target_prime = self.equation.initial_prime(x)

        return tf.reduce_sum((U - target) ** 2) + tf.reduce_sum((U_t - target_prime) ** 2)

    @tf.function
    def step(self):

        with tf.GradientTape() as tape:
            loss = self.boundary_condition_loss() + self.initial_condition_loss()

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss


    def optimize(self,minutes):
        ti0=time.time()
        OK=True

        while OK:
            OK=time.time()-ti0 < minutes*60

            print(f"optimize on period: {self.current_start} -> {self.current_end}")

            self.one_period()
            self.current_start+=self.delta_time/2
            self.current_end+=self.delta_time/2



    def one_period(self):

        go_on=True
        min_losses_of_this_period = []
        min_losses_place_of_this_period = []

        self.min_losses_by_period.append(min_losses_of_this_period)
        self.min_losses_place_by_period.append(min_losses_place_of_this_period)

        while go_on:
            self.step_count+=1

            loss = self.step().numpy()
            self.losses.append(loss)

            if len(min_losses_of_this_period)==0 or loss<min_losses_of_this_period[-1]:
                min_losses_of_this_period.append(loss)
                min_losses_place_of_this_period.append(self.step_count)
            else:
                if self.step_count-min_losses_place_of_this_period[-1] >self.patience:
                    break

            if self.step_count % 1000 == 0:
                clear_output(wait=True)
                plt.plot(self.losses)
                for places,values in zip(self.min_losses_place_by_period,self.min_losses_by_period):
                    plt.axvline(places[0],color="k")
                    plt.plot(places,values,".")

                plt.yscale("log")
                plt.show()
        #
        # clear_output(wait=True)
        # plt.plot(self.losses)
        # plt.yscale("log")
        # plt.show()


def test_learn_initial_condition():

    L = 2
    max_t = 0.1
    g1 = Grid(100, max_t, 100, L)
    equation = Equation(L)

    model = make_model(2)
    optimizer = tf.optimizers.Adam(1e-3)
    losses = []

    @tf.function
    def go():
        with tf.GradientTape() as tape:
            x = tf.random.uniform([100], 0, 2)
            t = tf.random.uniform([100], 0, 0.1)
            target = equation.initial(x)[:, tf.newaxis]
            tx = tf.stack([t, x], axis=1)
            loss = tf.reduce_sum((target - model(tx)) ** 2)

        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        return loss

    for _ in range(2000):
        loss = go().numpy()
        losses.append(loss)

    plt.plot(losses)
    plt.show()

    x = tf.linspace(tf.constant(0.), 2, 100)
    t = tf.zeros_like(x)
    tx = tf.stack([t, x], axis=1)
    plt.plot(x, model(tx))
    plt.plot(x, equation.initial(x))

    plt.show()


def test_learn_initial_condition_prime():

    L = 2
    model = make_model(1)
    optimizer = tf.optimizers.Adam(1e-3)
    losses = []

    @tf.function
    def step():
        with tf.GradientTape() as tape_w:
            #x = tf.random.uniform([100], 0, L)
            x = tf.linspace(tf.constant(0.),L,100)
            target = x#tf.ones_like(x)#x

            with tf.GradientTape() as tape_x:
                tape_x.watch(x)
                U = model(x)
            U_t = tape_x.gradient(U, x)

            loss = tf.reduce_sum((target - U_t) ** 2)+ U[0]**2

        gradient = tape_w.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        return loss


    for _ in range(1000):
        loss = step().numpy()
        losses.append(loss)

    plt.plot(losses)
    plt.show()

    x = tf.linspace(tf.constant(0.), 2, 100)
    plt.plot(x, model(x))

    plt.show()


def test_learn_initial_condition_prime_several():


    L = 2
    max_t=0.1
    model = make_model(2)
    optimizer = tf.optimizers.Adam(1e-3)
    losses = []


    @tf.function
    def go():
        x = tf.random.uniform([100], 0, L)
        t = tf.random.uniform([100], 0, max_t)

        with tf.GradientTape() as tape_w:

            with tf.GradientTape() as tape_x:
                tape_x.watch(x)
                tx=tf.stack([t,x],axis=1)
                U = model(tx)
            U_t = tape_x.gradient(U, x)

            #U=model(tx)[:,0]
            loss = tf.reduce_sum((U_t-x)**2)
        gradient = tape_w.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        return loss


    for _ in range(1000):
        loss = go().numpy()
        losses.append(loss)

    plt.plot(losses)
    plt.show()

    x = tf.linspace(tf.constant(0.), 2, 100)

    for t in tf.linspace(tf.constant(0.),max_t,10):
        ts = tf.ones_like(x)*t
        tx=tf.stack([ts,x],axis=1)
        plt.plot(x, model(tx))

    plt.show()


def test_agent():

    L = 2
    equation = Equation(L)

    agent=Agent(L,equation)
    agent.optimize(1)

    def u(t,x):
        tx=tf.stack([t,x],axis=1)
        return agent.model(tx)
    def u_t(t,x):
        with tf.GradientTape() as tape:
            tape.watch(t)
            U=u(t,x)
        U_t=tape.gradient(U,t)
        return U_t

    max_t = agent.current_end
    grid=Grid(100,max_t,100,L)


    U=u(grid.t_,grid.x_)
    U=grid.to_mat(U)
    #plt.title("x->u(t,x) for several t")
    grid.plot_several_times(U,equation.initial(grid.x))
    plt.show()

    U_t=u_t(grid.t_,grid.x_)
    U_t=grid.to_mat(U_t)
    #plt.title("x->u'(t,x) for several t")
    grid.plot_several_times(U_t,equation.initial_prime(grid.x))
    plt.show()

    plt.title("model solution")
    grid.plot_2d(U)
    plt.show()

    plt.title("analytic solution")
    U_ana=equation.u(grid.t_,grid.x_)
    U_ana=grid.to_mat(U_ana)
    grid.plot_2d(U_ana)
    plt.show()

    print("error:",tf.reduce_mean((U_ana-U)**2).numpy())




if __name__=="__main__":
    #observation()
    #test_checking()
    #test_learn_initial_condition()
    test_agent()
    #test_learn_initial_condition_prime_several()

    #test_learn_initial_condition_prime()
