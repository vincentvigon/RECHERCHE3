import numpy as np
import time
from typing import List,Dict,Set
from abc import ABC, abstractmethod
from collections import deque
pp=print
import copy
import matplotlib.pyplot as plt
from matplotlib import cm

#import inspect


print("Popup loaded!")
#todo la moyenne des meilleurs ne doit pas se faire avec des records trop éloigné
#problème: comment juger ???

#todo metric multi-dim

class History:

    def __init__(self):

        self.localTime_previous_cumulated_values = 0
        self.score_times =  []
        self.metrics_times:Dict[str,List[float]] = {}
        self.metrics_flags: Dict[str, Set[int]] = {}
        self.metrics_values:Dict[str,List[float]]={}
        self.localTime_isActive=False

    """Important: il faut démarrer le local time dès que l'agent est actif. Puis l'arrêter dès qu'il est inactif """
    def start_local_time(self):
        self.localTime_restart=time.time()
        assert not self.localTime_isActive , "on a oublié d'arrêter le localTime"
        self.localTime_isActive =True

    def stop_local_time(self):
        if not self.localTime_isActive:
            return
        self.localTime_previous_cumulated_values=self.get_local_time()
        self.localTime_isActive =False


    def get_local_time(self):
        if not self.localTime_isActive: #c'est logique
            return self.localTime_previous_cumulated_values
        else:
            return time.time()-self.localTime_restart+self.localTime_previous_cumulated_values

    def record_metric(self, name,value,addTag=False):
        assert is_number(value) ,f"Error when recording the value of '{name}': must be a number, but is: {value} of type {type(value)}"
        metric_times=self.metrics_times.get(name,[])
        metric_values=self.metrics_values.get(name,[])
        metric_times.append(self.get_local_time())
        metric_values.append(value)
        self.metrics_times[name]=metric_times
        self.metrics_values[name]=metric_values

        if addTag:
            metric_timeFlags=self.metrics_flags.get(name, set())
            metric_timeFlags.add(len(metric_times)-1)
            self.metrics_flags[name]=metric_timeFlags

import tensorflow as tf
def is_number(x):
    return isinstance(x, (int, float, np.float32, np.float64, np.int32, np.int64))

class Abstract_Agent(ABC):

    @abstractmethod
    def optimize_and_return_score(self)->float:
        pass

    @abstractmethod
    def get_famparams(self)->Dict[str,float]:
        pass

    @abstractmethod
    def set_and_perturb_famparams(self, famparam, period_count:int)->None:
        pass

    """ faut-il plutot demandé un dico à l'utilisateur ? """
    @abstractmethod
    def set_weights(self,weights:List):
        pass

    @abstractmethod
    def get_copy_of_weights(self)->List:
        pass

    #facultatif
    def perturb_famparams_on_decadence(self,period_count:int)->None:
        """
        Quand le score courrant de l'agent est dans les weak, mais que
        son record est dans les strong, les poids de l'agents sont ramener en
        arrières, et cette méthode est appelée
        :return:
        """
        pass

    def to_register_on_mutation(self)->Dict[str, float]:
        # by default an empty dictionnary
        return dict()

    def to_register_at_period_end(self)->Dict[str,float]:
        #by default an empty dictionnary
        return dict()

    def on_overfitting_restore_me(self)->bool:
        #exemple d'utilisation:
        # l'agent compare son score_train et son score_val. Quand le score_val est trop bas:
        # l'agent augmente son dropout ou sa pénalisation, puis la méthode "on_overfitting_restore_me"
        # renvoie "True", cela demande
        # au familyTrainer de restorer l'agent, en lui mettant la moyenne des poids des anciens records
        pass
    #
    # def return_score(self):
    #     #appeler  pour les tests ou la validation: pas d'optimisation
    #     pass


class Agent_wraper:

    def __init__(self, name, agent:Abstract_Agent, max_nb_best,nb_current_scores,current_score_fn):
        self.name=name
        self.agent=agent
        self.max_nb_best=max_nb_best
        self.current_score_fn=current_score_fn

        self.best_weights = deque(maxlen=max_nb_best)


        self.nb_consecutive_decadence=0

        self.decadent_score_and_weight=None

        self.best_score = None

        self.nb_current_scores=nb_current_scores
        self.current_scores = deque(maxlen=self.nb_current_scores)
        self.best_famparams: Dict[str, float]

        self.long_name=self.name

    #
    # def get_name_suffixed(self,decadence):
    #     res=self.name
    #     for mut in self.mutation_names:
    #         #la double flèche indique le transfert des poids et des famparams
    #         res+="⇇"+mut
    #     return res
    #
    # def perhapds_restablish_decadent(self):
    #     if self.decadent_score_and_weight is not None and self.decadent_score_and_weight[0]>self.best_score:
    #         print(f"Attention, le meilleurs score de l'agent {self.name} est décadent")
    #         self.best_score=self.decadent_score_and_weight[0]
    #         self.best_weights=self.decadent_score_and_weight[1]

    def save_at_best(self,new_best_score:float):
        self.best_weights.append(self.agent.get_copy_of_weights())
        self.best_score=new_best_score
        self.best_famparams=copy.deepcopy(self.agent.get_famparams())


    def mean_of_best_weights(self):
        nb_wei = len(self.best_weights[0])
        nb_best = len(self.best_weights)
        new_wei_list = []
        for j in range(nb_wei):
            shape = self.best_weights[0][j].shape
            res = np.zeros(shape)
            sum_ponde=0
            for i in range(nb_best):
                ponde=1/(nb_best-i+1)
                sum_ponde+=ponde
                res += self.best_weights[i][j]*ponde
            res /= sum_ponde
            new_wei_list.append(res)
        return new_wei_list

    def restore(self):
        new_wei_list = self.mean_of_best_weights()
        self.agent.set_weights(new_wei_list)
        #on ne change pas les famparams


    #exploitation
    def load_from_another(self, better_agent_w: 'Agent_wraper', period_count):

        #décadence
        # if decadence:
        #     """ L'agent a jadis optenu le meilleurs score, mais maintenant il est nul.
        #       On enregistre quand même les poids du temps de sa splendeur """
        #     if self.decadent_score_and_weight is None or self.decadent_score_and_weight[0]<self.best_score:
        #         self.decadent_score_and_weight=(self.best_score,self.best_weights)

        """les weights (ou learning variable) d'un réseau sont toujours une liste ou un tuple de tenseur"""
        new_weights = better_agent_w.mean_of_best_weights()

        self.long_name += "⇇" + better_agent_w.name
        self.agent.set_and_perturb_famparams(copy.deepcopy(better_agent_w.best_famparams), period_count)
        self.agent.set_weights(new_weights)

        """
         Ici, c'est pas l'idéal: on attribut comme current_score le best_score du l'other_agent.
         Or le mutant devrait faire ses preuves avant de passer dans la liste des premier.
         Mais on veut éviter qu'il ne soit remuter immédiatement après (il faudrait mettre en place un mécanisme de bonus pour les jeunes mutants)
         """
        self.current_scores=deque(maxlen=self.nb_current_scores)
        self.current_scores.append(better_agent_w.best_score)
        """ne pas oublier de vider la liste des best_weights. Les anciens peuvent être assez différents : le changement de famparams peu induire des changement brutal de poids (ex: coef de pénalisation)"""
        self.best_weights=deque(maxlen=self.max_nb_best)
        self.best_weights.append(new_weights)


    def load_from_myself(self, period_count):

        new_weights = self.mean_of_best_weights()
        self.long_name += "↫" + self.name

        if hasattr(self.agent, 'perturb_famparams_on_decadence'):
            self.agent.perturb_famparams_on_decadence(period_count)
        else:
            self.agent.set_and_perturb_famparams(copy.deepcopy(self.best_famparams), period_count)

        self.current_scores = deque(maxlen=self.nb_current_scores)
        self.current_scores.append(self.best_score)
        """ne pas oublier de vider la liste des best_weights. Les anciens peuvent être assez différents : le changement de famparams peu induire des changement brutal de poids (ex: coef de pénalisation)"""
        self.best_weights = deque(maxlen=self.max_nb_best)
        self.best_weights.append(new_weights)



    def compute_current_score(self):
        # en par défaut self.current_score_fn c'est la moyenne des 3 dernier scores courants
        return self.current_score_fn(self.current_scores)

    #
    # def load_from_another_old(self, better_agent_w: 'Agent_wraper', period_count, decadence: bool):
    #
    #
    #
    #     """les weights (ou learning variable) d'un réseau sont toujours une liste ou un tuple de tenseur"""
    #     new_weights = better_agent_w.mean_of_best_weights()
    #
    #     # décadence
    #     if decadence:
    #         self.long_name += "↫" + better_agent_w.name
    #         """cet enregistrement des best_weight décadent est inutile quand self.max_nb_best=1 """
    #         if self.decadent_score_and_weight is None or self.decadent_score_and_weight[0] < self.best_score:
    #             self.decadent_score_and_weight = (self.best_score, self.best_weights)
    #         if hasattr(self.agent, 'perturb_famparams_on_decadence'):
    #             self.agent.perturb_famparams_on_decadence(period_count)
    #         else:
    #             self.agent.set_and_perturb_famparams(copy.deepcopy(better_agent_w.best_famparams), period_count)
    #     # mutation normale
    #     else:
    #         self.long_name += "⇇" + better_agent_w.name
    #
    #         self.agent.set_and_perturb_famparams(copy.deepcopy(better_agent_w.best_famparams), period_count)
    #
    #     self.agent.set_weights(new_weights)
    #
    #     """
    #      Ici, c'est pas l'idéal: on attribut comme current_score le best_score du l'other_agent.
    #      Or le mutant devrait faire ses preuves avant de passer dans la liste des premier.
    #      Mais on veut éviter qu'il ne soit remuter immédiatement après (il faudrait mettre en place un mécanisme de bonus pour les jeunes mutants)
    #      """
    #     #self.current_score = better_agent_w.best_score
    #     self.current_scores = deque(maxlen=self.nb_current_scores)
    #     self.current_scores.append(better_agent_w.best_score)
    #     """ne pas oublier de vider la liste des best_weights. Les anciens peuvent être assez différents : le changement de famparams peu induire des changement brutal de poids (ex: coef de pénalisation)"""
    #     self.best_weights = deque(maxlen=self.max_nb_best)
    #     self.best_weights.append(new_weights)


def get_a_name(i:int):
    #les premiers noms sont des lettres, ensuite c'est a1,b1,c1,etc
    letters="a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
    letters=letters.split(" ")
    nb=len(letters)
    if i<nb:
        return letters[i]
    return letters[i%nb]+str(i//nb)

def transform_period_for_each(period_for_each):
    if period_for_each.endswith("s"):
        period_for_each = period_for_each[:-1]
    if period_for_each.endswith("e"):
        period_for_each = period_for_each[:-1]

    if period_for_each.endswith("second"):
        period_unity="second"
        period_duration=float(period_for_each[:-6].rstrip())

    elif period_for_each.endswith("minut"):
        period_unity = "second"
        period_duration = float(period_for_each[:-5].rstrip())*60

    elif period_for_each.endswith("step"):
        period_unity = "step"
        period_duration = float(period_for_each[:-5].rstrip())

    else: raise Exception("perdio must finish by second(e.s) or minut(e.s) or step(s)")

    return period_duration,period_unity


class Family_trainer:
    instance_count=0
    def __init__(self,
                 agents:List[Abstract_Agent],
                 # à la fin de chaque période on fait les mutations
                 period_duration:str, #ex: "10 steps", "20 seconds", "3 minute"
                 #nombre d'agent faible qui sont mutés à la fin de chaque période
                 nb_weak=1,
                 #nombre d'agent fort transmettant leur poids et le famparams
                 nb_strong=1,
                 nb_bestweights_averaged=1,
                 color="k",
                 nb_current_scores=3,
                 current_score_fn=np.mean,#np.max pour garder un agent qui renvoie un nan
                 max_nb_consecutive_decadence=2,
                 name=None,
                 ):

        Family_trainer.instance_count+=1

        self.period_duration,self.period_duration_unity=transform_period_for_each(period_duration)
        self.nb_current_scores=nb_current_scores
        self.current_score_fn=current_score_fn
        self.max_nb_consecutive_decadence=max_nb_consecutive_decadence
        self.name=name if name is not None else "fam_"+str(Family_trainer.instance_count)

        self.nb_weak=nb_weak
        if self.nb_weak>len(agents):
            print(f"Warning: attention, le nombre de strong:{self.nb_weak} est plus grand que le nombre d'agent:{len(agents)}, par conséquent il sera diminué à:{max(len(agents)//2,1)}.")
            self.nb_weak=max(len(agents)//2,1)

        self.nb_strong=nb_strong
        if self.nb_strong>len(agents):
            print(f"Warning: attention, le nombre de strong:{self.nb_strong} est plus grand que le nombre d'agent:{len(agents)}, par conséquent il sera diminué à:{len(agents)}.")
            self.nb_strong=len(agents)

        self.nb_bestweights_averaged=nb_bestweights_averaged

        self.color=color
        self._period_count=-1
        self._agent_pass_count=0

        #un dico car on voudrait pouvoir supprimer des agents
        self.agents:Dict[str,Agent_wraper] = {}
        self.history=History()

        for i,agent in enumerate(agents):
            agent_name=get_a_name(i)
            agent_w = Agent_wraper(agent_name, agent, self.nb_bestweights_averaged,self.nb_current_scores,self.current_score_fn)
            self.agents[agent_name] = agent_w
            for k, v in agent.get_famparams().items():
                self.history.record_metric(k, v)

        self.best_decadent_score=None
        self.best_decadent_weights=None
        self.best_decadent_famparams=None


    def interupt_period(self):
        self.history.stop_local_time()


    def period(self):

        self._period_count += 1
        agents_list=list(self.agents.values())
        if self._period_count==0:
            #tous les agents doivent passer au moins une fois avant de muter.
            print(f"\n{self.name},échauffement ", end="")
            for agent_w in agents_list:
                self.pass_one_agent(agent_w)
        else:
            self.history.start_local_time()
            print(f"\n{self.name},period:{self._period_count} ",end="")
            step_count = 0
            ok = True
            while ok:
                step_count += 1
                if self.period_duration_unity == "second":
                    #on cumule le temps. Avantage: si une pédiode dépasse (à cause d'un step un peu long), la suivante devra être plus courte.
                    #inconvénient: il ne faut pas changer la period_duration au milieu de l'apprentissage
                    #todo changer en utilisant un delta time
                    ok = self.history.get_local_time() < self.period_duration*self._period_count
                else:
                    ok = step_count < self.period_duration

                #si des agents passent pas lors d'une période, ils passeront dans la suivante.
                self.pass_one_agent(agents_list[self._agent_pass_count % len(agents_list)])
                self._agent_pass_count+=1


            #on arrête le localTime, car cela peut être le tour d'un autre familyTrainer
            self.history.stop_local_time()

            #a la fin de chaque période: mutation
            self.mutation()

    def pass_one_agent(self, agent_w:Agent_wraper):

        score=agent_w.agent.optimize_and_return_score()
        if np.isnan(score): score = -float("inf")

        self.history.record_metric("score", score)

        agent_w.current_scores.append(score)

        if  agent_w.best_score is None or score>agent_w.best_score:
            agent_w.save_at_best(score)
            print(agent_w.name+"↗"+str(np.round(score,4)),end="")
        else:
            print("-",end="")

        if hasattr(agent_w.agent,"to_register_at_period_end"):
            for k, v in agent_w.agent.to_register_at_period_end().items():

                self.history.record_metric(k, v)


        if hasattr(agent_w.agent, "on_overfitting_restore_me"):
            if agent_w.agent.on_overfitting_restore_me():
                # l'agent reprend ses anciens poids records
                agent_w.restore()

    def mutation(self):

        agents_best_score_sorted = sorted(self.agents.values(), key=lambda a_w: a_w.best_score)
        agents_current_score_sorted = sorted(self.agents.values(), key=lambda a_w: a_w.compute_current_score())

        """il est possible que la liste des strongs et la liste des weak aient des éléments en commum.
        notamment quand nb_weak+nb_strong > len(self.agents)
        Ce n'est pas génant, même une mutation d'un agent vers lui-même est intéressante.
        
        
        On dit qu'un agent est décadent quand il est dans les strong (classement relatif à son best score)
        mais aussi dans les weak (classement relatif à ses derniers score courants)
        et que son best score est sensiblement plus grand que son current_score (1.5 fois plus grand)
        Quand un agent est trop souvant décadent, on l'élimine (tout en l'enregistrant pour la fin car il a pu obtenir un score intéressant)
        """

        weaks=agents_current_score_sorted[:self.nb_weak]
        weak_names={weak.name for weak in weaks}

        decadent_names=[]
        decadents = []
        for i in range(self.nb_strong):
            strong=agents_best_score_sorted[-i]
            if strong.name in weak_names:
                sco=strong.compute_current_score()
                #ex:  sco=-10.  strong.best_score > -10+5
                #ex:  sco=+10.  strong.best_score > +10+5
                if strong.best_score>sco+np.abs(sco)*0.5:
                    decadent_names.append(strong.name)
                    decadents.append(strong)
                    print(f"\n/!\ L'agent:{strong.name} est décadent pour la {strong.nb_consecutive_decadence+1}-ième fois consécutive; record:{strong.best_score}, scores courants:{strong.current_scores}, best_famparams: {strong.best_famparams} ")

        strong_non_decadent=[]
        for strong in agents_best_score_sorted:
            if strong.name not in decadent_names:
                strong_non_decadent.append(strong)
        strongs=strong_non_decadent[-self.nb_strong:]
        if len(strongs)<self.nb_strong:
            print(f"ATTENTION: il y a seulement {len(strongs)} strong non-décadent, ce qui est inférieur au nombre de strong réclamé: {self.nb_strong}")

        print(", mutations:", end="")


        for weak in weaks:
            strong  = strongs[np.random.randint(len(strongs))]
            weak.load_from_another(strong,self._period_count)
            self.registration(weak,False)


        for decadent in decadents:
            if decadent.nb_consecutive_decadence<self.max_nb_consecutive_decadence:
                decadent.load_from_myself( self._period_count)
                decadent.nb_consecutive_decadence += 1
                self.registration(decadent, True)
            else:
                strong = strongs[np.random.randint(len(strongs))]
                #pour éventuellement récupérer ce décadent à la toute fin
                if self.best_decadent_score is None or strong.best_score>self.best_decadent_score:
                    self.best_decadent_score=strong.best_score
                    self.best_decadent_weights=decadent.best_weights
                    self.best_decadent_famparams=decadent.best_famparams

                decadent.nb_consecutive_decadence=0
                decadent.load_from_another(strong,self._period_count)
                self.registration(decadent, False)



    def registration(self,agent:Agent_wraper,mark:bool):
        for k, v in agent.agent.get_famparams().items():
            """on ajoute un tag pour repérer ces nouveaux poids """
            self.history.record_metric(k, v, False)

        if hasattr(agent.agent, "to_register_on_mutation"):
            for k, v in agent.agent.to_register_on_mutation().items():
                self.history.record_metric(k, v, mark)

        print(f"{agent.long_name}|", end="")



    def plot_metric(self, metric: str, ax=None, transformation=None):

        if ax is None:
            fig,ax=plt.subplots()

        ax.set_xlabel("local time")
        ax.set_ylabel(metric)

        x = self.history.metrics_times[metric]
        y = self.history.metrics_values[metric]
        if transformation is not None:
            y=transformation(y)

        flags=self.history.metrics_flags.get(metric,set())

        for i in range(len(x)):
            if i in flags:
                ax.scatter(x[i], y[i], color=self.color, marker = '+')
            else:
                #surtout ne pas mettre edgecolors="none". Cela rame
                ax.scatter(x[i], y[i], color=self.color, s=1)

    def plot_two_metrics(self,metric0:str,metric1:str,ax):

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(metric0)
        ax.set_ylabel(metric1)

        x = self.history.metrics_values[metric0]
        y = self.history.metrics_values[metric1]
        assert len(x)==len(y),f"les métricques {metric0} et {metric1} n'ont pas été enregistrée le même nombre de fois"

        flags0 = self.history.metrics_flags.get(metric0,set())
        flags1 = self.history.metrics_flags.get(metric0,set())
        flags=flags0.union(flags1)


        for i in range(len(x)):
            if i in flags:
                ax.scatter(x[i], y[i], color=self.color, marker='+',alpha=i / len(x))
            else:
                ax.scatter(x[i], y[i], color=self.color, s=1, alpha=i / len(x))

    def stats_of_best(self,nb_best=None):
        if nb_best is None:
            nb_best = int(len(self.agents) * 0.5)
        assert nb_best<=len(self.agents)

        agent_w_sorted=sorted(self.agents.values(),key=lambda ag:ag.best_score)
        best:List[Agent_wraper]=agent_w_sorted[-nb_best-1:]

        res=dict()
        sum_score=0
        for agent in best:
            sum_score+=agent.best_score
            for k,v in agent.best_famparams.items():
                res[k]=res.get(k,0)+v*agent.best_score

        for k,v in res.items():
            res[k]/=sum_score

        return res


    def get_best_agent(self, mean_its_weights=False, including_decadent=True)->Abstract_Agent:

        liste=sorted(self.agents.values(),key=lambda agent:agent.best_score)
        best=liste[-1]

        decadent_loaded=False
        if including_decadent:
            if self.best_decadent_score is not None and best.best_score<self.best_decadent_score:
                print("ATTENTION: le meilleurs agent est un agent décadent. Si voulez un non-décadent, utiliser including_decadent=False")
                #on récupère le pire agent pour lui transferer les poids du décadent
                an_agent=liste[0]
                an_agent.agent.set_weights(self.best_decadent_weights)
                an_agent.best_famparams=self.best_decadent_famparams
                best=an_agent
                decadent_loaded=True

        if mean_its_weights and not decadent_loaded:
            weights=best.mean_of_best_weights()
        else:
            weights=best.best_weights[-1]
        best.agent.set_weights(weights)

        return best.agent

def test_transform_period_for_each():
    assert transform_period_for_each("10seconds")==(10,"second")
    assert transform_period_for_each("1.2seconds")==(1.2,"second")
    assert transform_period_for_each("10 seconds")==(10,"second")
    assert transform_period_for_each("10 secondes")==(10,"second")
    assert transform_period_for_each("10 seconde")==(10,"second")
    assert transform_period_for_each("10 minutes")==(10*60,"second")
    assert transform_period_for_each("10.5 steps")==(10.5,"step")


def test_is_number():
    assert is_number(3)
    assert is_number(3.2)
    assert is_number(np.array([3],dtype=np.float32)[0])
    assert not is_number(np.array([3]))
    assert not is_number([3,4])
    assert is_number(np.nan)

class Agent_test(Abstract_Agent):

    def optimize_and_return_score(self) -> float:
        self.weights[0]+=1
        return np.random.uniform(0,2)+self.famparams["normal"]+self.famparams["decadence"]

    def get_famparams(self) -> Dict[str, float]:
        return self.famparams

    def set_and_perturb_famparams(self, famparam, period_count: int) -> None:
        self.famparams["normal"]+=1
    def perturb_famparams_on_decadence(self,period_count:int) ->None:
        self.famparams["decadence"] += 2

    def set_weights(self, weights: List):
        self.weights=weights

    def get_copy_of_weights(self) -> List:
        return self.weights.copy()

    def __init__(self):
        self.famparams={"normal":0,"decadence":0}
        self.weights=[np.ones([2])]

def test_decadence():

        agents=[Agent_test() for _ in range(3)]
        fm=Family_trainer(agents,"5 step",nb_bestweights_averaged=2,nb_strong=2)
        for _ in range(30):
            fm.period()
        fm.get_best_agent()


if __name__=="__main__":
    test_decadence()
    test_is_number()
    #test_transform_period_for_each()
