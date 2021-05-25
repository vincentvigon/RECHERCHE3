import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from typing import *

class Show_trajectory:
    def __init__(self, one_dim, env):
        self.one_dim = one_dim
        self.env:Bille_Env = env
        self.reset()

    def update(self, old_state):
        if self.one_dim:
            self.x.append(self.time)
            self.y.append(old_state[0])
        else:
            self.x.append(old_state[0])
            self.y.append(old_state[1])

        self.time += 1

    def reset(self):
        self.time = 0
        self.x = []
        self.y = []

    def show(self):
        fig, ax = plt.subplots()

        if self.one_dim:
            ax.set_xlabel("time")
            ax.set_ylabel("space")
            ax.set_ylim([self.env.state_lower_bounds[0], self.env.state_upper_bounds[0]])
        else:
            ax.set_xlabel("first coordinate")
            ax.set_ylabel("second coordinate")
            ax.set_xlim([self.env.state_lower_bounds[0], self.env.state_upper_bounds[0]])
            ax.set_ylim([self.env.state_lower_bounds[1], self.env.state_upper_bounds[1]])

        ax.plot(self.x[0], self.y[0], "o")
        ax.plot(self.x, self.y, ".-")
        plt.show()

class Bille_Env:

    def __init__(self, dimension, sigma=1):

        self.dimension = dimension
        self.dim_state=dimension
        self.dim_action=dimension

        self.sigma = sigma
        self.name = "Bille"

        self.state_lower_bounds=-10*np.ones([dimension])
        self.state_upper_bounds = +10 * np.ones([dimension])

        self.action_lower_bounds=-2*np.ones([dimension])
        self.action_upper_bounds = +2 * np.ones([dimension])

        # API gym
        self._max_episode_steps = 100

        self._do_render = False
        self.monitor = Show_trajectory(self.dimension == 1, self)

        self.reset()

    # API gym
    def reset(self):
        self.position = np.random.uniform(self.state_lower_bounds*0.2, self.state_upper_bounds*0.2, size=self.dimension)
        self.count = 0
        return self.position

    # API gym
    def step(self, action):
        self.count += 1

        assert len(action) == self.dimension, "bad dimension for action"
        for i in range(self.dimension):
            assert self.action_lower_bounds[i] <= action[i] <= self.action_upper_bounds[i], "action out of range"

        next_position = self.position + action + self.sigma * np.random.normal(size=self.dimension)

        terminal_bad = False
        terminal_good = False

        # on pert si la bille sort d'un rectangle
        for i in range(self.dimension):
            inside = self.state_lower_bounds[i] <= self.position[i] <= self.state_upper_bounds[i]
            if not inside:
                terminal_bad = True
                break

        if terminal_bad:
            reward = -10
        else:
            reward = 1

        # on gagne si la bille reste 500 fois
        if self.count > 500:
            terminal_good = True
            reward = 50

        self.position = next_position
        self.monitor.update(next_position)

        terminal = terminal_bad or terminal_good
        if terminal:
            self.reset()  # la position est réinitialiser
            if self._do_render:
                self.monitor.show()
                self.monitor.reset()

        return next_position, reward, terminal, {}

    # API gym
    def render(self):
        self._do_render = True

    def stop_render(self):
        self._do_render = False

def make_actor(
        dim_state:int,
        dim_action:int,
        lower_bounds:np.ndarray,
        upper_bounds:np.ndarray,
        model_struct:Dict[str,tuple]
):
    lower_bounds = tf.constant(lower_bounds, dtype=tf.float32)
    upper_bounds = tf.constant(upper_bounds, dtype=tf.float32)
    assert lower_bounds.shape == upper_bounds.shape == (dim_action,), "bad bounds"

    # state as input
    input_state = tf.keras.layers.Input([dim_state])

    input_out = input_state
    for i in model_struct["state_layer_dims"]:
        input_out = tf.keras.layers.Dense(i, activation="relu")(input_out)

    concat = input_out
    for i in model_struct["common_layer_dims"]:
        concat = tf.keras.layers.Dense(i, activation="relu")(concat)

    actor = concat
    for i in model_struct["actor_layer_dims"]:
        actor = tf.keras.layers.Dense(i, activation="relu")(actor)

    actor = tf.keras.layers.Dense(dim_action, activation="sigmoid")(actor)

    low = lower_bounds[tf.newaxis, :]
    up = upper_bounds[tf.newaxis, :]
    actor = actor * (up - low) + low

    model = tf.keras.Model(inputs=input_state, outputs=actor)

    return model


def make_critic(
        dim_state:int,
        dim_action:int,
        model_struct:Dict[str,tuple]
):
    state_input = tf.keras.layers.Input([dim_state])
    action_input = tf.keras.layers.Input([dim_action])

    state_out = state_input
    # state as input
    for i in model_struct["state_layer_dims"]:
        state_out = tf.keras.layers.Dense(i, activation="relu")(state_out)

    # Action as input
    action_out = action_input
    for i in model_struct["action_layer_dims"]:
        action_out = tf.keras.layers.Dense(i, activation="relu")(action_out)

    concat = tf.keras.layers.Concatenate()([state_out, action_out])
    for i in model_struct["common_layer_dims"]:
        concat = tf.keras.layers.Dense(i, activation="relu")(concat)

    critic = concat
    for i in model_struct["critic_layer_dims"]:
        critic = tf.keras.layers.Dense(i, activation="relu")(critic)

    critic = tf.keras.layers.Dense(1)(critic)

    model = tf.keras.Model(inputs=[state_input, action_input], outputs=critic)

    return model


make_actor(self.env.dim_state, self.env.dim_action, self.env.action_lower_bounds, self.env.action_upper_bounds, self.dense_struct)
make_critic(self.env.dim_state, self.env.dim_action, self.dense_struct)

def test_env(do_render=False):
    env=Bille_Env(2)
    num_states = env.dim_state
    print("dim of State Space ->  {}".format(num_states))
    num_actions = env.dim_action
    print("dim of Action Space ->  {}".format(num_actions))

    print("Max Value of Action ->  {}".format(env.action_upper_bounds))
    print("Min Value of Action ->  {}".format(env.action_lower_bounds))


    print("Max Value of State ->  {}".format(env.state_upper_bounds))
    print("Min Value of State ->  {}".format(env.state_lower_bounds))

    print("un état", env.reset())
    def random_action():
        return np.random.uniform(env.action_lower_bounds,env.action_upper_bounds,size=env.dim_action)

    print("step: \n next_state,reward,done,info \n", env.step(random_action()))

    if do_render:
        env.render()
    for i in range(100):
        next_state, reward, done, info = env.step(random_action())


default_model_struct={
                    "action_layer_dims":(16,32),
                    "state_layer_dims":(16,32),
                    "common_layer_dims":(64,64),
                    "critic_layer_dims":(64,),
                    "actor_layer_dims":(64,)
                    }


def test_critic():
    model = make_critic(2, 1,default_model_struct)
    for var in model.trainable_variables:
        print(var.shape, end="->")

    model.summary()
    state = tf.ones([50, 2])
    action = tf.ones([50, 1])
    print("result:", model([state, action]).shape)

def test_actor():
    model=make_actor(2,3,np.array([0,0,0]),np.array([10,10,10]),default_model_struct)
    for var in model.trainable_variables:
        print(var.shape,end="->")

    model.summary()
    state=np.random.uniform(-5,5,size=[50,2])
    action=np.random.uniform(-5,5,size=[50,3])
    res=model([state,action])
    print("result:",res.shape)
    print("bounds:",np.min(res),np.max(res))

def test_speed_of_models():

    batch_size=50 #même durée avec batch_size=1
    state=np.random.uniform(-5,5,size=[batch_size,2])
    action=np.random.uniform(-5,5,size=[batch_size,3])
    model=make_actor(2,3,np.array([0,0,0]),np.array([10,10,10]),default_model_struct)
    #la compilation n'accélère pas et ne sert à rien (car on va utiliser la GradientTape)
    #model.compile(optimizer=keras.optimizers.Adam(),loss="mse")

    @tf.function
    def eval_model():
        return model([state,action])

    t=time.time()
    for i in range(1000):
        eval_model()
    print("avec @tf.function",time.time()-t)


    def eval_model():
        return model([state,action])

    t=time.time()
    for i in range(1000):
        eval_model()
    print("sans @tf.function",time.time()-t)

    def eval_model():
        return model.predict([state,action])

    t=time.time()
    for i in range(100):
        eval_model()
    print("sans model.predict()",(time.time()-t)*10) #on multiplie par 10 et on fait seulement 100 itération


def all_tests():
    test_critic()
    test_actor()
    test_speed_of_models()
    test_env(False)

all_tests()



