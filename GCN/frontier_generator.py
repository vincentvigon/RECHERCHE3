
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from IPython.display import Image
import time
import json
import scipy.stats as stats
from skimage import feature


class Frontier:

    kind_iles= "kind_iles"
    kind_gaussian_noise="kind_gaussian_noise"
    kind_gamma_noise="kind_gamma_noise"
    kind_trivial="kind_trivial"

    frontier_smooth="frontier_smooth"
    frontier_oscilating="frontier_oscilating"
    frontier_ile="frontier_ile"


    def __init__(self,kind= "kind_iles",frontier_kind="frontier_smooth"):
        self.freqs = (1, 6)  # le nombre d'ile dans une largeur (=une hauteur). Défaut: entre 1 et 6
        self.perturbs = (0.02, 0.4)  # intensité des déformations. Défaut; entre 0.02 et 0.4
        self.kind= kind
        self.frontier_kind=frontier_kind
        self.sigma1=0.5
        self.sigma2=1
        self.alpha1=0.5
        self.alpha2=6


    def continuous_part_gamma_noise(self,a_shape,alpha):
        X = tf.random.gamma(a_shape, alpha=alpha)
        X-=stats.gamma.mean(a=alpha)
        X/=stats.gamma.std(a=alpha)
        return X

    def check_histo_of_noise_gamma(self):
        a_shape=(10_000,)
        X1=self.continuous_part_gamma_noise(a_shape,self.alpha1)
        X2=self.continuous_part_gamma_noise(a_shape,self.alpha2)

        plt.hist([X1,X2],bins=20)
        plt.show()


    def continuous_part_gaussian_noise(self,a_shape,sigma):
        return tf.random.normal(a_shape,stddev=sigma)

    def continuous_part_ile(self, a, b, batch_size):

        assert len(a.shape)==2, "cette fonction prend des batchs de vecteurs"

        nu0 = tf.random.uniform([batch_size,1],self.freqs[0], self.freqs[1])
        nu1 = tf.random.uniform([batch_size,1],self.freqs[0], self.freqs[1])

        per_coef0 = tf.random.uniform([batch_size,1],self.perturbs[0], self.perturbs[1])
        per_coef1 = tf.random.uniform([batch_size,1],self.perturbs[0], self.perturbs[1])

        per0 = tf.sin(2 * np.pi * a)
        per1 = tf.sin(2 * np.pi * b)

        a_ = a + per_coef0 * per0
        b_ = b + per_coef1 * per0 * per1

        y = tf.sin(np.pi * nu0 * a_) * tf.sin(np.pi * nu1 * b_)
        "et elle recrache des batchs de vecteur"
        return y


    @tf.function
    def compute(self, a, b, batch_size=None):
        assert a.shape == b.shape

        if batch_size is None:
            assert len(a.shape)==2,"si on ne précise pas de batch size, les entrées 'a' et 'b' doivent être déjà des batch de vecteur"
            batch_size=len(a) #la taille du batch c'est la longueur de a (et de b)
        else:
            assert len(a.shape)==1, "si on précise un batch_size, les entrées 'a' et 'b' doivent être des vecteurs"
            a = a[tf.newaxis,:]
            b = b[tf.newaxis,:]


        if self.kind==Frontier.kind_iles:
            cont0=self.continuous_part_ile(a, b, batch_size)
            cont1=self.continuous_part_ile(a, b, batch_size) #shape: batch_size * nb_data
        elif self.kind==Frontier.kind_gaussian_noise:
            cont0=self.continuous_part_gaussian_noise(a.shape,self.sigma1)
            cont1=self.continuous_part_gaussian_noise(a.shape,self.sigma2)
        elif self.kind==Frontier.kind_trivial:
            cont0=-1.
            cont1=+1.
        else: #self.kind==Ile_aleatoire.kind_gamma_noise:
            cont0=self.continuous_part_gamma_noise(a.shape,self.sigma1)
            cont1=self.continuous_part_gamma_noise(a.shape,self.sigma2)

        X,Y=self.discontinuous_part(a,b,cont0,cont1,batch_size)

        return X,Y


    def discontinuous_part(self,a, b,cont0,cont1,batch_size):
        assert len(a.shape)==2, "cette fonction prend en entrée des batchs de vecteurs"

        angle = tf.random.uniform([batch_size,1],-np.pi / 2, np.pi / 2)
        pente = tf.tan(angle)
        deb0 = tf.random.normal([batch_size,1],0.5,0.1)
        deb1 = tf.random.normal([batch_size,1],0.5,0.1)



        if self.frontier_kind==Frontier.frontier_smooth:
            power = tf.random.uniform([batch_size, 1], 0.2, 5)
            criterium= b < (a**power - deb0) * pente + deb1
        elif self.frontier_kind==Frontier.frontier_oscilating:
            nu = tf.random.uniform([batch_size, 1], 10, 50)
            ampl=tf.random.uniform([batch_size, 1], 0.005, 1)
            criterium= b-0.5 < tf.sin(nu*a)*ampl
        else:
            ile=self.continuous_part_ile(a,b,batch_size)
            criterium = ile>0

        y=tf.where(criterium, cont0, cont1)

        y_ref=tf.where(criterium, 0., 1.)

        return y,y_ref



def test_frontier():

    frontier=Frontier(kind=Frontier.kind_iles,frontier_kind=Frontier.frontier_oscilating)

    nb=100
    batch_size=25
    a = np.linspace(0., 1, nb,dtype=np.float32)
    aa, bb = tf.meshgrid(a, a)
    aaa = tf.reshape(aa,[-1])
    bbb = tf.reshape(bb,[-1])

    X,Y=frontier.compute(aaa, bbb, batch_size)

    print(X.shape)
    X=tf.reshape(X,[batch_size,nb,nb])
    Y=tf.reshape(Y,[batch_size,nb,nb])
    #
    # Y_front=np.zeros(Y.shape)
    #
    # for i,y in enumerate(Y):
    #     a=feature.canny(y.numpy())*1.
    #     Y_front[i,:,:]=a


    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    axs = axs.reshape(-1)

    for i,ax in enumerate(axs):
        ax.pcolormesh(aa,bb,X[i,:,:], cmap="jet",shading="gouraud")
        ax.contour(aa,bb,Y[i,:,:], colors='k')
        ax.axis("off")
    plt.show()



def random_diag(a, b):
    angle = np.random.uniform(-np.pi / 2, np.pi / 2)
    pente = np.tan(angle)
    deb0 = np.random.beta(3, 3)
    deb1 = np.random.beta(3, 3)

    a_=np.power(a,np.random.uniform(0.2,5))

    return np.where(b < (a_ - deb0) * pente + deb1, 0., 1)


def test_random_diag():
    a = np.linspace(0, 1, 100)
    aa, bb = np.meshgrid(a, a)

    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    axs = axs.reshape(-1)

    for ax in axs:
        ax.imshow(random_diag(aa, bb), cmap="jet", interpolation="bilinear")
        ax.axis("off")

    plt.show()

def test_tf_where():
    a=tf.linspace(-1,1,10)
    val0=tf.ones([10])
    val1=tf.zeros([10])
    res=tf.where(a>0,val0,val1)
    print(res)


def test_gamma():
    X=tf.random.gamma([1000],alpha=4)
    print("std",np.std(X))
    print(stats.gamma.std(a=4))
    print("mean",np.mean(X))
    print(stats.gamma.mean(a=4))

    frontier=Frontier()
    frontier.check_histo_of_noise_gamma()

    #X=frontier.continuous_part_gamma_noise([10,50],3)
    print(np.std(X))
    print(np.mean(X))

#
# def discontinuous_part():
#     batch_size=13
#     a=tf.ones([batch_size,100])
#     b=tf.ones([batch_size,100])
#     ab=tf.stack([a,b],axis=2)
#     print(ab.shape)
#
#
#     T = tf.random.uniform([batch_size, 2, 2])
#     """
#     ordre b,n,
#     T_ab[b,n,j]=sum_i  T[b,i,j] ab[b,n,i]   """
#     Tab=
#     print(Tab.shape)



if __name__=="__main__":
    #test_ile_perturb()
    #test_random_diag()
    #test_tf_where()
    #test_gamma()
    test_frontier()





