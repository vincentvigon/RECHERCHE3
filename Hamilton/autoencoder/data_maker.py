import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3,suppress=True,linewidth=10000)


#
# class Data_maker_curve:
#
#     def __init__(self,dim_data):
#         self.dim_data=dim_data
#
#
#     def make(self,size):
#         theta=tf.random.uniform(shape=[size],minval=0,maxval=1)
#         #theta=tf.cast(tf.linspace(-10.1234,10,1000),tf.float32)
#         data=[]
#
#         for i in range(self.dim_data):
#             data.append(tf.cos(theta*(i+1)*np.pi)*0.2)
#
#         return tf.stack(data,axis=1)
#
#     def make_sorted(self,size):
#         theta=tf.linspace(0,1,size)
#         theta=tf.cast(theta,dtype=tf.float32)
#         data=[]
#         for i in range(self.dim_data):
#             data.append(tf.cos(theta*(i+1)*np.pi)*0.2)
#
#         return tf.stack(data,axis=1)



class Data_maker_curve_periodic:

    def __init__(self,dim_data,domain_size,periodic,max_theta=1,min_max_normalization=False):
        self.dim_data=dim_data
        self.domain_size=domain_size
        self.periodic=periodic
        self.max_theta=max_theta
        self.min_max_normalization=min_max_normalization

        tf.random.set_seed(12134)
        self.fourier_coef=tf.random.uniform([dim_data,3],-1,1)

        self.freqs=np.random.randint(2,16,size=dim_data)


    def a_function(self,theta):
        theta=theta[:,np.newaxis]
        freq=self.freqs[np.newaxis,:]
        freq1=tf.cos(theta*freq)*self.fourier_coef[:,0][np.newaxis,:]
        freq2 = tf.cos(theta * (freq + 1) ) * self.fourier_coef[:,1][np.newaxis,:]
        freq3 = tf.cos(theta * (freq/2)) * self.fourier_coef[:,2][np.newaxis,:]

        if self.periodic:
            freqs=(freq1+freq2+freq3)*self.domain_size*3
            #on produit des données centrées
            return tf.math.floormod(freqs,self.domain_size)-self.domain_size/2
        else:
            freqs=(freq1 + freq2 + freq3)
            if self.min_max_normalization:
                mini=tf.reduce_min(freqs)
                maxi=tf.reduce_max(freqs)
                ampl=(maxi-mini)
                freqs-=mini
                freqs/=ampl
            return  freqs* self.domain_size-self.domain_size/2
    #
    # def a_function_old(self,theta,i):
    #     freq=self.freqs[i]
    #     freq1=tf.cos(theta*freq)*self.fourier_coef[freq,0]
    #     freq2 = tf.cos(theta * (freq + 1) ) * self.fourier_coef[freq,1]
    #     freq3 = tf.cos(theta * (freq//2)) * self.fourier_coef[freq,2]
    #
    #     if self.periodic:
    #         freqs=(freq1+freq2+freq3)*self.domain_size*3
    #         #on produit des données centrées
    #         return tf.math.floormod(freqs,self.domain_size)-self.domain_size/2
    #     else:
    #         freqs=(freq1 + freq2 + freq3)
    #         if self.min_max_normalization:
    #             mini=tf.reduce_min(freqs)
    #             maxi=tf.reduce_max(freqs)
    #             ampl=(maxi-mini)
    #             freqs-=mini
    #             freqs/=ampl
    #         return  freqs* self.domain_size-self.domain_size/2


    def make(self,size):
        theta=tf.random.uniform(shape=[size],minval=0,maxval=self.max_theta)
        return self.a_function(theta)

    def make_sorted(self,size):
        theta=tf.cast(tf.linspace(0,self.max_theta,size),tf.float32)
        return self.a_function(theta)



import time
def test_time_Data_maker_curve_periodic():
    data_maker=Data_maker_curve_periodic(1000,periodic=False,domain_size=1)
    ti0=time.time()
    X=data_maker.make(256)
    print(X.shape)
    print(time.time()-ti0)





def present_data(data, pred=None, nb=5, max_data=-1):
    if nb > data.shape[1]:
        nb = data.shape[1]

    fig, axs = plt.subplots(nb, nb, figsize=(10, 10), sharex="all", sharey="all")
    for i in range(nb):
        for j in range(nb):
            if j >= i:
                i_, j_ = i, j
            else:
                i_, j_ = -i, -j
            axs[i, j].scatter(data[:max_data, i_], data[:max_data, j_], s=1)
            if pred is not None:
                axs[i, j].scatter(pred[:max_data, i_], pred[:max_data, j_], s=1)


def present_data_margin(data, pred=None,max_dim=16):
    nb = min(data.shape[1],max_dim)
    fig, axs = plt.subplots(nb, 1,figsize=(10, nb), sharex="all")
    indices=np.arange(len(data))
    for i in range(nb):
        axs[i].scatter(indices,data[:, i],s=1)
        if pred is not None:
            axs[i].scatter(indices,pred[:, i],s=1)



def present_norm_of_data(data, pred, proj):
    norm = tf.norm(data, axis=1)
    norm_pred = tf.norm(pred, axis=1)
    norm_proj = tf.norm(proj, axis=1)
    abs_val = range(len(data))

    fix, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6))
    ax0.plot(abs_val, norm, label="data")
    ax0.plot(abs_val, norm_pred, label="pred")
    ax1.plot(abs_val, norm_proj, label="proj")
    ax0.legend()
    ax1.legend()

#
# def test_curve():
#     DIM_INPUT = 16
#
#     curve = Data_maker_curve(DIM_INPUT)
#     data=curve.make_sorted(300)
#     print(data.shape)
#     present_data_margin(data)
#     present_data(data)
#
#     plt.show()


def test_curve_periodic():
    DIM_INPUT = 16
    domain_size=0.2

    curve = Data_maker_curve_periodic(DIM_INPUT,domain_size,periodic=True)
    data=curve.make_sorted(300)
    print(data.shape)
    present_data_margin(data)
    present_data(data)

    plt.show()


if __name__=="__main__":
    #test_curve_periodic()
    test_time_Data_maker_curve_periodic()




