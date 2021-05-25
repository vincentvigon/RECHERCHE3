import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import *
import time
import ddqn_lib.ddqn as dd

class Toy_Env(dd.Abstract_Environment):
    def get_dim_state(self) -> int:
        return 1

    def get_dim_action(self) -> int:
        return 1

    def __init__(self,sigma=1):
        self.value=0
        self.count = 0
        self.sigma=sigma

    def reset(self)->np.ndarray:
        self.value=np.random.uniform(-10,10)
        self.count = 0
        return np.array([self.value])

    def step(self, action):
        self.value+= action + self.sigma * np.random.normal()
        self.count+=1

        terminal_bad = False
        terminal_good = False

        inside = -20 <= self.value <= +20
        if not inside:
            terminal_bad = True

        if terminal_bad:
            reward = -10
        else:
            reward = 1

        # on gagne si la bille reste 100 fois
        if self.count > 100:
            terminal_good = True
            reward = 50

        terminal = terminal_bad or terminal_good
        if terminal:
            self.reset()  # la position est réinitialiser

        return np.array([self.value]), reward, terminal


def actor_maker_fn():
    input_state=tf.keras.layers.Input([1])
    y=tf.keras.layers.Dense(4,activation="relu")(input_state)
    output_action=tf.keras.layers.Dense(1)(y)
    return tf.keras.Model(inputs=input_state,outputs=output_action)

def critic_maker_fn():
    input_state = tf.keras.layers.Input([1])
    input_action = tf.keras.layers.Input([1])
    y_state = tf.keras.layers.Dense(3,activation="relu")(input_state)
    y_action = tf.keras.layers.Dense(3,activation="relu")(input_action)
    y=tf.keras.layers.Concatenate()([y_state,y_action])
    y=tf.keras.layers.Dense(3,activation="relu")(y)
    output_critic = tf.keras.layers.Dense(1)(y)
    return tf.keras.Model(inputs=[input_state,input_action], outputs=output_critic)


""" Attention, parfois cela échoue complétement, et parfois cela réussit très très vite.
Exemple de réussite rapide:
scores
|1.2_record|-4.4|-5.1|8.8_record|13.5_record|77.0_record|150.0_record|150.0_record|150.0_record|112.5|150.0_record|150.0_record|108.5|150.0_record|92.0|150.0_record|150.0_record|86.0|115.3|150.0_recordvalidation:
scores validation:
|78.0|150.0|65.0|150.0|150.0|62.0|150.0|150.0|150.0|150.0
"""
def main():

    agent=dd.Agent_ddqn(Toy_Env(),actor_maker_fn,critic_maker_fn)
    scores=[]
    ite=-1
    print("scores")
    for _ in range(20):
        ite+=1
        score=agent.optimize_and_return_score()
        print(f"|{score:.1f}",end="")
        scores.append(score)
        if score>=np.max(scores):
            print("_record",end="")
            agent.save_actor_weights()
    fig,ax=plt.subplots()
    ax.set_title("score train")
    ax.plot(scores)

    print("validation:")
    agent.set_saved_actor_weights()
    print("scores validation:")
    for _ in range(10):
        ite+=1
        score=agent.valid_return_score()
        print(f"|{score:.1f}",end="")
        scores.append(score)

    fig, ax = plt.subplots()
    ax.set_title("score val")
    ax.plot(scores)
    plt.show()


main()