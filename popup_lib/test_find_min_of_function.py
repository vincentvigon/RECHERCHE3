from popup_lib.popup import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def loss_fn(x:tf.Variable,y:tf.Variable):
    difficulty=0.5 # avec 0.1 l'optimisation devient très facile
    norm1=tf.abs(x)+tf.abs(y)
    return 3*norm1+ difficulty * (1 - tf.sin( x * y * 500))


class Minimizer_agent(Abstract_Agent):

    def __init__(self, custom_set_perturb_famparams):
        #on veut pouvoir créer des agents avec des méthodes de perturbation différentes
        Minimizer_agent.set_and_perturb_famparams=custom_set_perturb_famparams

        self.weight0 = tf.Variable(np.random.uniform(-1,1))
        self.weight1 = tf.Variable(np.random.uniform(-1,1))
        self.famparams={"dir0":np.random.uniform(0,1e-1),"dir1":np.random.uniform(0,1e-1)}

    def get_famparams(self):
        return self.famparams

    def set_famparams(self, dico):
        self.famparams=dico



    def set_and_perturb_famparams(self, famparam, period_count: int) -> None:
        raise Exception("to be defined in the constructor")


    def optimize_and_return_score(self):

        with tf.GradientTape() as tape:
            loss=loss_fn(self.weight0,self.weight1)
        gradients=tape.gradient(loss,[self.weight0,self.weight1])

        self.weight0.assign_sub(gradients[0] * self.famparams["dir0"])
        self.weight1.assign_sub(gradients[1] * self.famparams["dir1"])

        return -loss.numpy()


    def set_weights(self, weights):
        self.weight0.assign(weights[0])
        self.weight1.assign(weights[1])

    def get_copy_of_weights(self):
        return [self.weight0.numpy(),self.weight1.numpy()]

    def to_register_on_mutation(self) ->Dict[str, float]:
        return {"wei0":self.weight0.numpy(),"wei1":self.weight1.numpy()}


def plot_loss_func(ax,r):

    x = np.linspace(-r, r, 100,dtype=np.float32)
    y = np.linspace(-r, r, 100,dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    Q = loss_fn(xx, yy)  ##1.2 - (xx ** 2 + yy ** 2)

    ax.set_aspect('equal')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.contourf(xx, yy, Q, 50, cmap='autumn')

def main_by_hand():
    agent=Minimizer_agent(None)
    agent.set_famparams({"dir0":0.01,"dir1":0.01})
    scores=[]
    weights0=[]
    weights1 = []
    for _ in range(20):
        score=agent.optimize_and_return_score()
        w0,w1=agent.get_copy_of_weights()
        weights0.append(w0)
        weights1.append(w1)
        scores.append(score)

    fig,ax=plt.subplots(1,1,figsize=(4,4))

    ax.set_title("scores")
    ax.plot(scores,".-")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_loss_func(ax,1)
    ax.set_title("weights")
    ax.plot(weights0,weights1,"k.")
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

    plt.show()

#
# def test_af():
#     class Bou:
#         def __init__(self,meth,name):
#             Bou.meth=meth
#             self.name=name
#
#
#         def meth(self):
#             pass
#
#     bou1=Bou(lambda self:print(self.name+"_1"),"bou1")
#     bou2=Bou(lambda self:print(self.name+"_2"),"bou2")
#     bou1.meth()
#     bou2.meth()
#     bou1.meth()
#     bou2.meth()


def main():

    def perturb_quiet(agent:Minimizer_agent,famparams,turn_count):
        agent.famparams=famparams
        agent.famparams["dir0"] *= np.random.uniform(0.8, 1.2)
        agent.famparams["dir1"] *= np.random.uniform(0.8, 1.2)

    def perturb_exited(agent: Minimizer_agent,famparams,turn_count):
        agent.famparams=famparams
        agent.famparams["dir0"] *= np.random.uniform(0.5, 1.5)
        agent.famparams["dir1"] *= np.random.uniform(0.5, 1.5)

    popsize = 10
    quiet_agents = [Minimizer_agent(perturb_quiet) for _ in range(popsize)]
    exited_agents = [Minimizer_agent(perturb_exited) for _ in range(popsize)]


    fam_trainers=[
                  Family_trainer(quiet_agents,
                                 period_duration="10 steps",
                                 color="g"
                                 ),
                  Family_trainer(exited_agents,
                                 period_duration="10 steps",
                                 color="k"
                                 )
                  ]
    try:
        for i in range(20):
            for fm in fam_trainers:fm.period()

    except KeyboardInterrupt:
        print("interuption")

    fig,ax=plt.subplots()
    ax.set_ylim(0,0.2)
    ax.set_xlim(0,0.2)
    for fm in fam_trainers:fm.plot_two_metrics("dir0","dir1",ax)


    fig,ax=plt.subplots()
    r=15
    plot_loss_func(ax,r)
    ax.set_ylim(-r, r)
    ax.set_xlim(-r, r)
    for fm in fam_trainers:
        fm.plot_two_metrics("wei0", "wei1",ax)


    fig, ax = plt.subplots()
    ax.set_ylim(-15, 0)
    for fm in fam_trainers:
        fm.plot_metric("score", ax)

    plt.show()


if __name__=="__main__":
    main()

