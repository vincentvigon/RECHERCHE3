
from popup_lib.popup import *

import numpy as np
import matplotlib.pyplot as plt

"""unitary test"""

class Agent_ultra_toy(Abstract_Agent):

    def get_famparams(self):
        return self.famparams

    def set_famparams(self, dico):
        self.famparams=dico

    def perturb_famparams(self):
        if np.random.random()<0.5:
            self.famparams["add0"]+=1
        else:
            self.famparams["add1"]-=1

    def __init__(self):
        self.wei0=np.array([0.])
        self.wei1=np.array([0.])
        self.famparams={"add0":0,"add1":0}

    def optimize_and_return_score(self) -> float:
        #c'est pas vraiment une optimization ici
        self.wei0 +=self.famparams["add0"]
        self.wei1 +=self.famparams["add1"]
        return self.return_score()

    def return_score(self)-> float:
        return self.wei0[0] + self.wei1[0]

    def set_weights(self, weights):
        self.wei0,self.wei1=weights

    def get_copy_of_weights(self):
        return self.wei0,self.wei1

    def to_register_on_mutation(self) ->Dict[str, float]:
        return {"wei0":self.wei0[0],"wei1":self.wei1[0]}


def main():
    agents=[Agent_ultra_toy(), Agent_ultra_toy()]
    family_trainer=Family_trainer(agents)

    for _ in range(10):
        family_trainer.period()

    family_trainer.plot_metric("score")
    family_trainer.plot_metric("add0")
    family_trainer.plot_metric("add1")

    family_trainer.plot_metric("wei0")
    family_trainer.plot_metric("wei1")

    print("\nstats_of_best:",family_trainer.stats_of_best())


    ###Test
    best_agent=family_trainer.get_best_agent()
    scores=[]
    for _ in range(5):
        scores.append(best_agent.return_score())
    fig,ax=plt.subplots()
    ax.set_title("best agent test")
    ax.plot(scores)
    plt.show()





if __name__=="__main__":
    #test_is_number()
    main()
