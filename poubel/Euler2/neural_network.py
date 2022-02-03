import tensorflow as tf
import tensorflow.keras.layers as layers



class Model(tf.keras.Model):

    def __init__(self,input_dim,odd_shrinkage:bool):
        super().__init__()
        d=64
        kernel_size=3
        nb_conv_layer=1
        dense_struct=(64,64,32)

        #attention, changer le shrinkage si vous modifier la structure du réseau
        self.lay=[]
        for _ in range(nb_conv_layer):
            self.lay.append(layers.Conv1D(d,kernel_size,activation="relu"))

        self.shrinkage = (kernel_size - 1) * nb_conv_layer

        if (odd_shrinkage and self.shrinkage%2==0) or (not odd_shrinkage and self.shrinkage%2==1):
            self.lay.append(layers.Conv1D(d, 4, activation="relu"))
            self.shrinkage+=3
        else:
            self.lay.append(layers.Conv1D(d, 3, activation="relu"))
            self.shrinkage+=2

        for units in dense_struct:
            self.lay.append(layers.Dense(units,activation="relu"))

        #Le résultat initial est 0
        self.lay.append(layers.Dense(1,kernel_initializer="zeros"))


        print("modèle crée avec un schrinkage de:", self.shrinkage)

        #pour obligé les modèle à créer les variables, sinon on a une exception: "variables created on a non first call"
        #c'est un peu couillon, car cela nous oblige à connaitre la taille de l'augmentation (=2 dans le cadre de Burger)
        self.call(tf.zeros([1,100,input_dim]))


    def call(self,X):
        for layer in self.lay:
            X=layer(X)
        return X



def test():

    def one(odd_shrinkage,n):
        model=Model(7,odd_shrinkage)

        @tf.function
        def accelerate():
            Y=model.call(tf.zeros([1,n,7]))
            shrinkage=n-Y.shape[1]
            if odd_shrinkage:
                assert shrinkage%2==1
            else:
                assert shrinkage%2==0
        accelerate()

    one(True,40)
    one(True,41)
    one(False,40)
    one(False,41)




if __name__=="__main__":
    test()