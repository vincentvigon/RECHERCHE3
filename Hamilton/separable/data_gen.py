import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


"""
cas  général
H(q,p)  est quelconque

Le schéma symplectique est implicite
q_n+1= q_n + dt d/dp H(q_n+1,p_n)
p_n+1= p_n - dt d/dq H(q_n+1,p_n)

-------------------------------------

cas dynamique séparable: H est de la forme
H(q,p) = H1(q) + H1(p)
d/dp H(q,p) = dH2(p) 
d/dq H(q,p) = dH1(q)

Le schéma symplectique est explicite
q_n+1= q_n + dt dH2(p_n)
p_n+1= p_n - dt dH1(q_n+1)

---------------------------------------

cas mécanique classique
H(q,p) = H1(q) + p^2/2 
d/dp H(q,p) = p 
d/dq H(q,p) = dH1(q)

Le schéma symplectique est explicite
q_n+1= q_n + dt p_n
p_n+1= p_n - dt dH1(q_n+1)
"""

class Param:

    config_independant="independant"
    config_mean_recall="mean_recall"
    config_sinus_perturb= "sinus_perturb"
    config_attractive_particles="attractive_particles"
    config_repulsive_particles="repulsive_particles"


    def __init__(self,config="independant",
                 nb_particle=12,
                 nb_t=1000,
                 batch_size=512,
                 sort_q0=True,
                 external_recall=0.5,
                 distrib_on_q=True
                 ):

        self.config=config
        self.nb_particle=nb_particle
        self.nb_t=nb_t
        self.batch_size = batch_size
        self.sort_q0=sort_q0
        self.external_recall=external_recall
        self.distrib_on_q=distrib_on_q

        """
        dq = +dH2(p)  (= p en méca classique)
        dp = -dH1(p)
        """
        if config=="independant":
            #resorts indépendants
            self.dH1 = lambda q:q
            self.dH2 = lambda p:p
        elif config=="mean_recall":
            #la force de rappel est la moyenne des particules
            self.dH1 = lambda q: self.external_recall*q + tf.ones([q.shape[0],self.nb_particle])*tf.reduce_mean(q,axis=1)[:,tf.newaxis]
            self.dH2 = lambda p: p
        elif config=="sinus_perturb":
            #un peu n'importe quoi
            self.dH1 = lambda q: self.external_recall*q+tf.sin(tf.reduce_mean(q,axis=1)[:,tf.newaxis])
            self.dH2 = lambda p: p+tf.sin(tf.reduce_mean(p,axis=1)[:,tf.newaxis])
        elif config=="attractive_particles":
            #les particules s'attirent les unes au autres. La force d'attraction décroit avec la distance
            """dp_i  = - sum_j |p_i-p_j|"""
            self.dH1 = lambda q: self.external_recall*q + tf.reduce_sum(q[:,:,tf.newaxis] - q[:,tf.newaxis,:],axis=2)
            self.dH2 = lambda p: p

        elif config=="repulsive_particles":
            #les particules s'attirent les unes au autres. La force d'attraction décroit avec la distance
            """dp_i  = - sum_j |p_i-p_j|"""
            self.dH1 = lambda q: self.external_recall*q - 0.99e-2*tf.reduce_sum(q[:,:,tf.newaxis] - q[:,tf.newaxis,:],axis=2)
            self.dH2 = lambda p: p

        else:
            raise Exception("whats this config:"+self.config)

        self.dt=0.01



    #des configurations aléatoires pour le train
    def initial_part_distri_train(self):
        q0 = tf.random.uniform([self.batch_size,self.nb_particle],0.,0.5) #tf.linspace(0.2, 1, self.nb_particle)
        # Si on n'utilise pas de compression symétrique: on peut ordonner les particules
        # cela améliore les choses car le modèle n'apprend pas toutes les permutations possibles.
        # L'amélioration est surtout visible en grande dimension. Ex: dim 20, config mixe, on passe d'une erreur de 4 à un erreur de 0.4 en 1000 itérations
        # Attention1: il faut que les jeux tests et val soient aussi ordonnés
        # Attention2: si on part avec une distributions initiale (position, vitesse) non indépendantes,
        #il faudrait permuter les vitesses avec la même permutation que celle utilisée pour les positions
        if self.sort_q0:
            q0 = tf.sort(q0,axis=1)

        p0 = tf.zeros([self.batch_size,self.nb_particle])
        if self.distrib_on_q:
            return q0, p0
        else:
            return p0, q0


    def initial_part_distri_val(self):
        q0= tf.linspace(0.01,0.48,self.nb_particle)[tf.newaxis,:]
        if self.sort_q0:
            q0 = tf.sort(q0,axis=1)
        p0=tf.zeros([1,self.nb_particle])
        if self.distrib_on_q:
            return q0,p0
        else:
            return p0,q0

    #une distribution légérement différente pour le test
    def initial_part_distri_test(self):
        q0= tf.linspace(0.03,0.45,self.nb_particle)[tf.newaxis,:]
        if self.sort_q0:
            q0 = tf.sort(q0,axis=1)
        p0=tf.zeros([1,self.nb_particle])
        if self.distrib_on_q:
            return q0, p0
        else:
            return p0, q0



@tf.function
def solver_symplectic_separable(nb_t:int, dt:float, q0, p0, dH1, dH2):
    #print(f"'tf_symplectic_separable' est tracé avec les arguments primitids: nb_t:{nb_t}, dt:{dt}")
    #print(f"    les tenseurs: q0:{q0.shape}, q0:{p0.shape}, et les fonctions dH1 et dH2")

    assert q0.shape == p0.shape

    qs=tf.TensorArray(tf.float32, size=nb_t, element_shape=q0.shape, dynamic_size=False, clear_after_read=True)
    ps=tf.TensorArray(tf.float32, size=nb_t, element_shape=q0.shape, dynamic_size=False, clear_after_read=True)

    for t in tf.range(nb_t):
        q0 += dt * dH2(p0)
        p0 -= dt * dH1(q0)

        qs=qs.write(t, q0)
        ps=ps.write(t, p0)
    return qs.stack(),ps.stack()




def visu(qs, ps):

    fig, axs = plt.subplots( figsize=(10,10))
    for i in range(ps.shape[1]):
        axs.plot( qs[:,i],ps[:,i])
    axs.set_title("Phase portrait p(q)", fontsize = 14)
    axs.set_xlabel("q")
    axs.set_ylabel("p")
    axs.set_aspect("equal")

    plt.show()



def test_configs():

    param=Param(Param.config_attractive_particles)
    param.external_recall=1
    param.nb_particle=10
    param.distrib_on_q=True

    q0,p0=param.initial_part_distri_val()
    qs,ps=solver_symplectic_separable(param.nb_t, param.dt, q0, p0, param.dH1, param.dH2)

    print(qs.shape)
    print(ps.shape)

    visu(qs[:,0,:],ps[:,0,:])


if __name__=="__main__":
    test_configs()
