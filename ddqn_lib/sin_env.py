import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
pp=print
import ddqn_lib.ddqn as dd
import popup_lib.popup as pop

DIM=300
SIGMA_NOISE=3

def random_sinus_sum(dim):
    res=np.zeros([dim])
    x=np.linspace(0,2*np.pi,dim)
    for i in range(1,10):
        res+=np.random.uniform(-0.2,0.2)*np.sin(i*x)
    return res

"""Il faut bien régler les paramètres sigma et initial_size
 pour que l'on ne gagne pas à chaque coup: utiliser le test_zero_action() pour vérifier """
class Sinus_Env(dd.Abstract_Environment):

    def __init__(self, dim=DIM, sigma_noise=SIGMA_NOISE, initial_size=5, reward_when_win=100):
        self.dim=dim
        self.sigma_noise = sigma_noise
        self.initial_size=initial_size
        self.reward_when_win=reward_when_win
        self.value=np.zeros([dim])
        self.count = 0
        self.render_is_active=False

    def get_dim_state(self) -> int:
        return self.dim

    def get_dim_action(self) -> int:
        return self.dim

    def reset(self)->np.ndarray:
        self.value= random_sinus_sum(self.dim)*self.initial_size
        self.count=0
        return self.value

    def step(self, action):
        assert action.shape==(self.dim,), f"action must have the dimension {self.dim} but shape is {action.shape} "
        self.value+= action + self.sigma_noise * random_sinus_sum(self.dim)
        self.count+=1

        terminal_bad = False
        terminal_good = False

        too_big=np.sum(self.value>20)
        too_small=np.sum(self.value<-20)
        inside = too_big+too_small ==0
        if not inside:
            terminal_bad = True

        if terminal_bad:
            reward = -self.reward_when_win
        else:
            reward = 1

        # on gagne si la fonction reste 100 fois dans les bornes
        if self.count > 200:
            terminal_good = True
            reward = self.reward_when_win

        terminal = terminal_bad or terminal_good
        if terminal:
            self.reset()  # la position est réinitialiser
        if self.render_is_active:
            self._record_for_render()

        return self.value, reward, terminal


    def start_render(self):
        self.render_is_active=True
        self.curves=[]
        self.current_render_stop=0

    def do_render(self):
        x=np.linspace(0,2*np.pi,self.dim)
        self.render_is_active=False
        nb=len(self.curves)
        for i,curve in enumerate(self.curves):
            if i==0:
                alpha=1
                color="r"
            else:
                alpha = i / nb
                color="k"
            plt.plot(x,curve,color,alpha=alpha)
        plt.show()

    def _record_for_render(self):
        self.curves.append(self.value.copy())

def evaluate_policy(env:Sinus_Env, policy):
    rewards=[]
    ep_lengths=[]
    nb_win=0
    nb_test=40
    for _ in range(nb_test):
        r = None
        ep_length = 0
        cum_reward=0
        done=False
        s=env.reset()
        while not done:
            ep_length+=1
            s,r,done=env.step(policy(s))
            cum_reward+=r
        if r==env.reward_when_win:
            nb_win+=1
        ep_lengths.append(ep_length)
        rewards.append(cum_reward)

    print(f"win:{nb_win/nb_test} cum_rewards:{np.mean(rewards)}, episode length:{np.mean(ep_lengths)}")

def test_zero_policy():
    env=Sinus_Env(DIM)
    policy=lambda state:np.zeros([env.dim])
    evaluate_policy(env, policy)
#test_zero_policy()

def test_env_graph():
    dim=300
    env=Sinus_Env(dim)
    zero_action=np.zeros([dim])
    done=False
    env.start_render()
    r=0
    count=0
    while not done:
        count+=1
        s_,r,done=env.step(zero_action)
    env.do_render()
    print(f"reward:{r}, count:{count}")

def layer_triple(y):
    y = tf.keras.layers.Conv1D(6, 5, activation="relu", padding="same")(y)
    y = tf.keras.layers.Conv1D(6, 5, activation="relu", padding="same")(y)
    y = tf.keras.layers.Conv1D(6, 5, activation="relu", padding="same")(y)
    return y

def actor_maker_fn():
    input_state=tf.keras.layers.Input([DIM,1])
    y=layer_triple(input_state)
    output_action=tf.keras.layers.Conv1D(1,5,padding="same")(y)
    #il faut une action de taille comparable au bruit pour pouvoir le compenser
    output_action=output_action[:,:,0]*SIGMA_NOISE
    return tf.keras.Model(inputs=input_state,outputs=output_action)

def critic_maker_fn():
    input_state = tf.keras.layers.Input([DIM,1])
    input_action = tf.keras.layers.Input([DIM,1])
    y_state = layer_triple(input_state)
    y_action = layer_triple(input_action)
    y=tf.keras.layers.Concatenate()([y_state,y_action])
    y=layer_triple(y)
    output_critic = tf.keras.layers.Conv1D(1,5,padding="same")(y)
    output_critic = output_critic[:,:,0]
    return tf.keras.Model(inputs=[input_state,input_action], outputs=output_critic)

def test_models():
    batch_size=3
    input_action=np.random.normal(size=[batch_size,DIM,1])
    input_critic = np.random.normal(size=[batch_size,DIM,1])
    model_actor=actor_maker_fn()
    model_critic=critic_maker_fn()
    res_actor=model_actor(input_action).numpy()
    print(f"actor result shape{res_actor.shape} , sdt:{np.std(res_actor.flatten())}")
    res_critic=model_critic([input_action,input_critic]).numpy()
    print(f"actor result shape{res_critic.shape} , sdt:{np.std(res_critic.flatten())}")

#test_models()

def main_one():
    env=Sinus_Env()
    print("The zero policy gives:")
    evaluate_policy(env, lambda state: np.zeros([env.dim]))

    agent=dd.Agent_ddqn(env,actor_maker_fn,critic_maker_fn,
                     perturb_action_sigma=1e-2,
                     min_seconds_before_score=3)
    scores=[]
    ite=-1
    minutes=3
    ti0=time.time()
    print("score and std*10000")
    try:
        while time.time()-ti0<minutes*60:
            ite+=1
            score=agent.optimize_and_return_score()
            print(f"|{score:.0f},{agent.std*100000:.2f}",end="")
            scores.append(score)
            if score >= np.max(scores):
                print("_record", end="")
    except KeyboardInterrupt:
        print("\ninteruption")

    plt.show()

    #todo: ploter des couples state,politique(state) pour comprendre

def main():
    env = Sinus_Env()
    print("The zero policy gives:")
    evaluate_policy(env, lambda state: np.zeros([env.dim]))

    env = Sinus_Env()
    print("The zero policy gives:")
    evaluate_policy(env, lambda state: np.zeros([env.dim]))

    def perturb_famparams(famparams):
        famparams["lr"]*=np.random.uniform(0.7,1.7)
        famparams["perturb_action_sigma"]*=np.random.uniform(0.7,1.7)
        gamma=np.clip(famparams["gamma"]*np.random.uniform(0.9,1.1),0.5,0.999)
        famparams["gamma"]=gamma


    agents = [dd.Agent_ddqn(env, actor_maker_fn, critic_maker_fn,perturb_famparams,
                          perturb_action_sigma=10**np.random.uniform(-3,-1),
                          min_seconds_before_score=2)
              for _ in range(5)]

    family_trainer=pop.Family_trainer(agents,"5 seconds")

    for _ in range(10):
        family_trainer.period()




if __name__=="__main__":
    main()
