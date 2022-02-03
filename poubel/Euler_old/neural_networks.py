import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from typing import *

#import pandas as pd
np.set_printoptions(precision=3, linewidth=100000)
pp=print

from Euler.backend import K
import Euler.core_solver as core
from Euler.core_solver import Param

k_tf=K("tf",32)

#=================================================
layers=tf.keras.layers
model_struct_default=(32,64,32)
regularizer=tf.keras.regularizers.l2(1)


def pad_L_R_for_model(W, pad:int, param:Param):

    if param.BC_model == Param.BC_neumann:
        W = core.neumann_padding(W, k_tf, pad)
    elif param.BC_model == Param.BC_periodic:
        W = core.periodic_padding(W, k_tf, pad)
    elif param.BC_model == Param.BC_reflexive:
        W = core.reflexive_padding(W, k_tf, pad,False)
    else:
        raise Exception("unknown padding:"+param.BC_model)
    return W


def unrol(W_roll,window_length):
    return W_roll[:,:,window_length//2,:]



def shift_eye(window_size,dim_features,shift):
    res=np.zeros([window_size,window_size*dim_features])
    res[:,shift*window_size:(shift+1)*window_size]=np.eye(window_size)
    return res


def create_tricy_convo(window_size,dim_features):

    def initializer(shape,dtype=None):
        tricky_kernel=np.zeros([window_size,dim_features,window_size*dim_features])
        for j in range(dim_features):
            tricky_kernel[:,j,:]=shift_eye(window_size,dim_features,j)
        return tricky_kernel

    layer= layers.Conv1D(window_size*dim_features,window_size,kernel_initializer=initializer)

    return layer


class Difference_model_several_convo:

    def __init__(self, param: Param, kernel_size=4,conv_struct=(64,32,32),dense_struct=(32,16), return_zeros=True):

        self.param = param
        self.kernel_size = kernel_size
        self.dense_struct = dense_struct
        self.conv_struct=conv_struct

        assert (kernel_size-1)*len(conv_struct) % 2 == 1, f"(kernel_size-1)*len(conv_struct)=({kernel_size}-1)*{len(conv_struct)} must be odd"

        self.return_zeros=return_zeros
        self.dense_layers = []
        self.conv_layers=[]

        for i in self.conv_struct:
            layer=layers.Conv1D(i,self.kernel_size,activation="relu")
            self.conv_layers.append(layer)

        for i in self.dense_struct:
            layer = layers.Dense(i, activation="relu")
            self.dense_layers.append(layer)

        if self.return_zeros:
            final_layer = layers.Dense(3 ,kernel_initializer="zeros", bias_initializer="zeros")
        else:
            final_layer = layers.Dense(3)

        dim_features = param.augmentation_dim

        #Le +1 c'est pour extraire les décallages

        total_padding=(kernel_size-1)*len(conv_struct)+1
        self.pad_each = total_padding // 2

        input_ = layers.Input([param.nx_coarse+total_padding, dim_features])  # remplacer None par un entier (ex: 20) pour faire des test où l'on voit les shapes

        input_L = input_[:, :-1, :]
        input_R = input_[:, 1:, :]

        current_L = input_L
        current_R = input_R

        for layer in self.conv_layers:
            current_L= layer(current_L)
        for layer in self.dense_layers:
            current_L = layer(current_L)
        end_L = final_layer(current_L)

        for layer in self.conv_layers:
            current_R=layer(current_R)
        for layer in self.dense_layers:
            current_R = layer(current_R)
        end_R = final_layer(current_R)

        end = end_R - end_L

        self.keras_model = tf.keras.Model(inputs=input_, outputs=end)

    @tf.function
    def __call__(self, X):
        X = pad_L_R_for_model(X, self.pad_each, self.param)
        return self.keras_model(X)


class Difference_model_convo:

    def __init__(self, param: Param, window_size,censure=False   , model_struct=model_struct_default,return_zeros=True):
        assert window_size % 2 == 1

        self.param = param
        self.window_size = window_size
        self.censure=censure
        self.model_struct = model_struct
        self.return_zeros=return_zeros
        self.hiden_layers = []

        for i in self.model_struct:
            layer = layers.Dense(i, activation="relu")
            self.hiden_layers.append(layer)

        if self.return_zeros:
            final_layer = layers.Dense(3 ,kernel_initializer="zeros", bias_initializer="zeros")
        else:
            final_layer = layers.Dense(3)

        dim_features = param.augmentation_dim
        input_ = layers.Input([param.nx_coarse+window_size-1, dim_features])  # remplacer None par un entier (ex: 20) pour faire des test où l'on voit les shapes

        input_L = input_[:, :-1, :]
        input_R = input_[:, 1:, :]

        #Le 60 est arbritraire. Pour ressembler au modèle tricky, il faudrait prendre window_size*dim_augmentation
        normal_convo=layers.Conv1D(60,window_size-1)

        input_L = normal_convo(input_L)
        input_R = normal_convo(input_R)

        current_L = input_L
        current_R = input_R

        for layer in self.hiden_layers:
            current_L = layer(current_L)
        end_L = final_layer(current_L)

        for layer in self.hiden_layers:
            current_R = layer(current_R)
        end_R = final_layer(current_R)

        end = end_R - end_L

        if self.censure:
            censure = input_
            for _ in range(window_size // 2):
                censure = layers.Conv1D(30, 3, activation="relu")(censure)
            censure = layers.Dense(20, activation="relu")(censure)
            censure = layers.Dense(1, activation="sigmoid")(censure)

            end=end*censure


        self.keras_model = tf.keras.Model(inputs=input_, outputs=end)

    def __call__(self, X):
        X = pad_L_R_for_model(X, self.window_size // 2, self.param)
        return self.keras_model(X)


class Difference_model_tricky:

    #cela peut être un famparam
    def toggle_tricky_kernel_trainable(self,yes):
        self.tricky_convo.trainable=yes

    def __init__(self, param: Param, window_size,censure=False   , model_struct=model_struct_default,return_zeros=True):
        assert window_size % 2 == 1

        self.param = param
        self.window_size = window_size
        self.censure=censure
        self.model_struct = model_struct
        self.return_zeros=return_zeros
        self.hiden_layers = []

        for i in self.model_struct:
            layer = layers.Dense(i, activation="relu")
            self.hiden_layers.append(layer)

        if self.return_zeros:
            final_layer = layers.Dense(3 ,kernel_initializer="zeros", bias_initializer="zeros")
        else:
            final_layer = layers.Dense(3)

        dim_features = param.augmentation_dim
        input_ = layers.Input([param.nx_coarse+window_size-1, dim_features])  # remplacer None par un entier (ex: 20) pour faire des test où l'on voit les shapes

        input_L = input_[:, :-1, :]
        input_R = input_[:, 1:, :]

        self.tricky_convo = create_tricy_convo(window_size - 1, dim_features)
        self.tricky_convo.trainable=False

        input_L = self.tricky_convo(input_L)
        input_R = self.tricky_convo(input_R)

        current_L = input_L
        current_R = input_R

        for layer in self.hiden_layers:
            current_L = layer(current_L)
        end_L = final_layer(current_L)

        for layer in self.hiden_layers:
            current_R = layer(current_R)
        end_R = final_layer(current_R)

        end = end_R - end_L

        if self.censure:
            censure = input_
            for _ in range(window_size // 2):
                censure = layers.Conv1D(30, 3, activation="relu")(censure)
            censure = layers.Dense(20, activation="relu")(censure)
            censure = layers.Dense(1, activation="sigmoid")(censure)

            end=end*censure


        self.keras_model = tf.keras.Model(inputs=input_, outputs=end)

    def __call__(self, X):
        X = pad_L_R_for_model(X, self.window_size // 2, self.param)
        return self.keras_model(X)


#
# class Difference_model_addi_old:
#
#     def __init__(self,param: Param,window_size,twice_the_same=True, model_struct=model_struct_default):
#         assert window_size%2==1
#
#         self.param=param
#         self.window_size=window_size
#         self.model_struct=model_struct
#         self.twice_the_same=twice_the_same
#         self.keras_model:tf.keras.Model=self._make_model()
#
#
#     def _model_one_part(self):
#         #self.window_size-1 car le résultat est une différence de 2 fenètre décallées
#         input_=layers.Input([self.window_size-1,self.param.augmentation_dim])
#         out_ = layers.Flatten()(input_)
#         pp("self.model_struct",self.model_struct)
#         for i in self.model_struct:
#             pp("->i",i)
#             out_ = layers.Dense(i, activation="relu",kernel_regularizer=regularizer)(out_)
#         end_=layers.Dense(3,kernel_regularizer=regularizer,kernel_initializer="zeros",bias_initializer="zeros")(out_)
#         return tf.keras.Model(inputs=input_,outputs=end_)
#
#     def _make_model(self):
#         input_= layers.Input([self.window_size,self.param.augmentation_dim])
#         input_L=input_[:,:self.window_size-1,:]
#         input_R=input_[:,1:,:]
#
#         model_L=self._model_one_part()
#         end_L=model_L(input_L)
#
#         if self.twice_the_same:
#             model_R=model_L
#         else:
#             model_R=self._model_one_part()
#         end_R=model_R(input_R)
#
#         end=end_R-end_L
#         model=tf.keras.Model(inputs=input_,outputs=end)
#         return model
#
#     def __call__(self,w):
#         return call_for_model_looking_window(w, self.keras_model, self.window_size,self.param)


#
# def test_rol_human():
#     data = np.arange(0, 7, 1)
#     w = np.stack([data, data], axis=1)
#     w = w[np.newaxis,:,:]
#     window_size=5
#     W_roll=pad_and_roll(w,window_size,Param(),True)
#     print("w.shape:",w.shape)
#     print("W_roll.shape:",W_roll.shape)
#     print("unrol(W_roll)-w:\n",unrol(W_roll,window_size)-w)
#     print("w:\n",w)
#     print("W_roll\n",W_roll)
#
# def test_rol_auto(roll_via_stacking:bool):
#     w = np.random.normal(size=[2, 7, 3])
#     window_size=7
#     W_roll=pad_and_roll(w,window_size,Param(),roll_via_stacking)
#     diff=unrol(W_roll,window_size)-w
#     assert tf.reduce_sum(tf.abs(diff))<1e-6

def test_model_convo():
    param = Param(nx=500)
    print("param.nx_coarse:", param.nx_coarse)
    window_size = 5
    model = Difference_model_convo(param, window_size, censure=False)
    w_coarse = np.ones([7, param.nx_coarse, param.augmentation_dim])
    Y = model(w_coarse)
    print(Y.shape)


def test_model_several_convo():
    param = Param(nx=500)
    print("param.nx_coarse:", param.nx_coarse)

    model = Difference_model_several_convo(param)
    w_coarse = np.ones([7, param.nx_coarse, param.augmentation_dim])



    def call_model(w_coarse):
        return model(w_coarse)


    for _ in range(5):
        Y=call_model(w_coarse)
        print(Y.shape)



def test_model():
    param=Param(nx=500)
    print("param.nx_coarse:",param.nx_coarse)
    window_size=5

    def censure_or_not(yes):
        model=Difference_model_tricky(param,window_size,censure=yes)

        model.toggle_tricky_kernel_trainable(True)
        nb_trainable=len(model.keras_model.trainable_variables)
        print("nb_trainable",nb_trainable)

        model.toggle_tricky_kernel_trainable(False)
        nb_trainable = len(model.keras_model.trainable_variables)
        print("nb_trainable", nb_trainable)

        w_coarse=np.ones([7,param.nx_coarse,param.augmentation_dim])
        Y=model(w_coarse)
        print(Y.shape)

    print("with censure")
    censure_or_not(True)
    print("without censure")
    censure_or_not(False)


if __name__=="__main__":
    #exploration()
    #test_model_convo()
    test_model_several_convo()
    #test_model_addi()
    #test_rol_auto()
    #test_penalization()

    #test_model()
    #test_rol()
