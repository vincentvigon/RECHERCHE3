import numpy as np
import tensorflow as tf
import time
from typing import *
from abc import ABC, abstractmethod
import popup_lib.popup as pop
pp=print

"""
Quand on update la target à chaque pas de temps, ce @tf.function divise la durée d'un époque par 2.
La moyennation des poids a un coup non negligeable.
A chaque fois que l'on change tau, tensorflow doit "retracer" la fonction et il met un warning.
donc: ne pas trop changer tau !   
"""
#todo préciser la signature du @tf.function

@tf.function
def update_target(target_model, model,tau):
    target_weights=target_model.trainable_variables
    weights=model.trainable_variables
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

#la fonction d'avant, pour tau=1
def transfer_weights_to_target(target_model, model):
    target_weights=target_model.trainable_variables
    weights=model.trainable_variables
    for (a, b) in zip(target_weights, weights):
        a.assign(b)


"""
Voilà ce que doit satisfaire un environnement pour interagir avec l'agent DDQN
"""
class Abstract_Environment(ABC):

    @abstractmethod
    def reset(self)->np.ndarray:
        pass

    @abstractmethod
    def step(self, action)->Tuple[np.ndarray,float,bool]: #next_state,reward,terminal
        pass

    @abstractmethod
    def get_dim_state(self)->int:
        pass

    @abstractmethod
    def get_dim_action(self)->int:
        pass


class Agent_ddqn(pop.Abstract_Agent):

    def __init__(self,
                 env:Abstract_Environment,
                 actor_maker_fn,
                 critic_maker_fn,
                 # le temps minimum d'attente avant  retourner 1 score (moyénné sur les épisode)
                 # C'est uniquement pour le family_trainer qui préfère des scores moyenné (pour l'early stopping)
                 # si None ou 0: on attend juste un épisode (qui peut être très court) avant de donner un sco
                 min_seconds_before_score=0,

                 #ci-dessous ce sont des famparams (suseptible d'être modifié par un family_trainer)
                 buffer_capacity=5000,
                 batch_size=64,
                 gamma=0.99,  # Discount factor
                 lr=1e-3,  # learning rate
                 perturb_action_sigma=1e-2,
                 perturb_action_decrease=0.995,
                 target_update_interval=1,
                 target_update_tau=0.01
                 ):

        self.env:Abstract_Environment = env
        self.actor_maker_fn=actor_maker_fn
        self.critic_maker_fn=critic_maker_fn
        self.min_seconds_before_score=min_seconds_before_score

        self.famparams = {
                "buffer_capacity":buffer_capacity,
                "batch_size": batch_size,
                "gamma": gamma,
                "lr": lr,
                "perturb_action_sigma":perturb_action_sigma,
                "perturb_action_decrease":perturb_action_decrease,
                "target_update_interval":target_update_interval,
                "target_update_tau":target_update_tau
                }

        self.initialize_models()
        self.critic_optimizer = tf.keras.optimizers.Adam(self.famparams["lr"])
        self.actor_optimizer = tf.keras.optimizers.Adam(self.famparams["lr"])

        self.buffer_counter = 0

        bc=self.famparams["buffer_capacity"]
        self.state_buffer = np.zeros((bc, self.env.get_dim_state()))
        self.action_buffer = np.zeros((bc, self.env.get_dim_action()))
        self.reward_buffer = np.zeros((bc, 1))
        self.next_state_buffer = np.zeros((bc, self.env.get_dim_state()))
        self.global_ite_count = 0


    # popup API
    def get_famparams(self):
        return self.famparams

    def initialize_models(self):

        self.actor = self.actor_maker_fn()
        self.critic = self.critic_maker_fn()

        self.nb_actor_wei=len(self.actor.get_weights())

        self.target_actor:tf.keras.Model = self.actor_maker_fn()
        self.target_critic:tf.keras.Model = self.critic_maker_fn()

        self.randomized_actor:tf.keras.Model = self.actor_maker_fn()

        transfer_weights_to_target(self.target_actor, self.actor)
        transfer_weights_to_target(self.target_critic, self.critic)

    # popup API
    def get_copy_of_weights(self) -> List:
        wei_actor=self.actor.get_weights()
        wei_critic=self.critic.get_weights()
        return wei_actor+wei_critic

    # popup API
    def set_weights(self, weights: List):
        self.actor.set_weights(weights[:self.nb_actor_wei])
        self.critic.set_weights(weights[self.nb_actor_wei:])

    #  (s,a,r,s') = (state,action,reward,next_state)
    def record(self, s, a, r, s_):
        # le modulo permet de remplacer les anciens enregistremenets
        index = self.buffer_counter % self.famparams["buffer_capacity"]
        self.state_buffer[index] = s
        self.action_buffer[index] = a
        self.reward_buffer[index] = r
        self.next_state_buffer[index] = s_
        self.buffer_counter += 1

    #todo @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):

        # Entrainement du critique
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            # on veut que le critique vérifie de plus en plus bellman
            y = reward_batch + self.famparams["gamma"] * self.target_critic([next_state_batch, target_actions])
            critic_value = self.critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # Entrainement de l'acteur
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            critic_value = self.critic([state_batch, actions])
            # L'acteur veut maximiser la valeur de son action donnée par le critique.
            # Pour maximiser on met un signe -
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        # apply_gradients suit l'opposé des gradients (il cherche le minimum)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )


    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.famparams["buffer_capacity"])
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.famparams["batch_size"])

        # Convert to tensors
        state_batch = tf.constant(self.state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.constant(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.constant(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.constant(self.next_state_buffer[batch_indices], dtype=tf.float32)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def policy(self, state, val_mode:bool):
        state_batch = tf.constant(state, dtype=tf.float32)[tf.newaxis, :]

        if val_mode:
            return self.target_actor(state_batch)[0]
        else:
            rate=self.famparams["perturb_action_decrease"]
            self.std=self.famparams["perturb_action_sigma"]* rate**self.global_ite_count

            transfer_weights_to_target(self.randomized_actor, self.actor)

            for var in self.randomized_actor.trainable_variables:
                noise=tf.random.normal(var.shape, stddev=self.std)
                var.assign_add(noise)

            return self.randomized_actor(state_batch)[0]

    # popup API
    def _run_one_episode(self, val_mode:bool):
        episodic_reward = 0
        prev_state = self.env.reset()
        self.initial_time = time.time()

        # attention, l'environnnement ne doit pas renvoyer d'épisode de longueur infini
        # amélioration (pas forcement)  interompre le train dès que le temps est dépassé, même si l'épisode n'est pas fini
        done = False
        while not done:  # 1 épisode
            self.global_ite_count += 1

            action = self.policy(prev_state,val_mode)
            state, reward, done = self.env.step(action)
            episodic_reward += reward

            if not val_mode:
                self.record(prev_state, action, reward, state)
                self.learn()
                # target_update est une paire (tau,update_interval)
                if  self.global_ite_count % self.famparams["target_update_interval"] == 0:
                    update_target(self.target_actor, self.actor, self.famparams["target_update_tau"])
                    update_target(self.target_critic, self.critic,self.famparams["target_update_tau"])

            prev_state = state

        return episodic_reward

    #popup API
    def optimize_and_return_score(self):
        ti0=time.time()
        score=0
        nb_ite=0
        finish=False
        while not finish:
            nb_ite += 1
            score+= self._run_one_episode(False)
            finish=self.min_seconds_before_score is None \
                   or  time.time()-ti0> self.min_seconds_before_score

        return score/nb_ite


    #API popup (il faut mettre du contenu si que le Family Trainer fasse quelque chose)
    def set_and_perturb_famparams(self, famparams, period_count: int):
        pass

    # popup API, facultatif
    def return_score(self) -> float:
        return self._run_one_episode(True)














