
import popup_lib.popup as pop
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict



class Agent_ultra_toy(pop.Abstract_Agent):

    def __init__(self):
        self.wei0=np.array([0.])
        self.wei1=np.array([0.])
        self.famparams={"add0":0,"add1":0}

    #Abstract_Agent: obligatoire
    def get_famparams(self):
        return self.famparams

    #Abstract_Agent: obligatoire
    def set_and_perturb_famparams(self,famparams,period_count):
        self.famparams=famparams

        if np.random.random()<0.5:
            self.famparams["add0"]+=1
        else:
            self.famparams["add1"]-=1

    #Abstract_Agent: obligatoire
    def optimize_and_return_score(self) -> float:
        #c'est pas vraiment une optimization ici
        self.wei0 +=self.famparams["add0"]
        self.wei1 +=self.famparams["add1"]
        return self.return_score()

    #Abstract_Agent: obligatoire
    def set_weights(self, weights):
        self.wei0,self.wei1=weights

    #Abstract_Agent: obligatoire
    def get_copy_of_weights(self):
        return self.wei0,self.wei1


    #Abstract_Agent: facultatif: pour ajouter des métriques
    #def to_register_at_period_end(self) ->Dict[str,float]:
    #je ne l'ai pas implémenter pour cet agent

    #Abstract_Agent: facultatif: pour faire des tests
    def return_score(self)-> float:
        return self.wei0[0] + self.wei1[0]


    #Abstract_Agent: facultatif: pour observer les poids
    def to_register_at_period_end(self) ->Dict[str,float]:
        return {"wei0":self.wei0[0],"wei1":self.wei1[0]}



def main():
    agents = [Agent_ultra_toy(), Agent_ultra_toy()]
    family_trainer = pop.Family_trainer(agents, period_duration="10 steps", nb_strong=1)

    for _ in range(20):
        family_trainer.period()

    family_trainer.plot_metric("score")
    family_trainer.plot_metric("add0")
    family_trainer.plot_metric("add1")

    family_trainer.plot_metric("wei0")
    family_trainer.plot_metric("wei1")

    plt.show()


main()