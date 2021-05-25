import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import Hamilton.autoencoder.agents as agents
import Hamilton.autoencoder.data_maker as dat
import Hamilton.autoencoder.utilities as util

np.set_printoptions(precision=3,suppress=True,linewidth=10000)


class LAE:

    def __init__(self, input_dim, struct_enc, reduce_dim, mix=True):
        self.input_dim=input_dim
        self.struct_enc=struct_enc
        self.reduce_dim=reduce_dim


        struct_dec=[(b,a) for (a,b) in  struct_enc[::-1]]

        X_in=tf.keras.layers.Input([input_dim])

        successive_feat_dim=[input_dim]
        X=X_in


        for (fan_in,fan_out) in struct_enc:
            X=self.block(X,fan_in,fan_out,mix)
            successive_feat_dim.append(X.shape[1])

        nb_fea_after_sparse=X.shape[1]

        X_reduced=tf.keras.layers.Dense(self.reduce_dim, activation="tanh")(X)
        successive_feat_dim.append(X_reduced.shape[1])

        self.encoder=tf.keras.Model(inputs=X_in,outputs=X_reduced)

        Y_in = tf.keras.layers.Input([self.reduce_dim])

        Y=tf.keras.layers.Dense(nb_fea_after_sparse,activation="tanh")(Y_in)
        successive_feat_dim.append(Y.shape[1])


        for (fan_in,fan_out) in struct_dec:
            Y=self.block(Y,fan_in,fan_out,mix)
            successive_feat_dim.append(Y.shape[1])

        self.decoder=tf.keras.Model(inputs=Y_in, outputs=Y)

        input_compo=tf.keras.Input([input_dim])
        self.compo=tf.keras.Model(inputs=input_compo,outputs=self.decoder(self.encoder(input_compo)))

        print("successive_feat_dim:",successive_feat_dim)


    def block(self,X, fan_in, fan_out , mix):

        nb_feature=X.shape[1]

        assert nb_feature % fan_in == 0

        #X0 = tf.reshape(X, [-1 , fan_in])

        X0 = tf.reshape(X, [-1, nb_feature // fan_in , fan_in])

        X1=tf.keras.layers.Dense(fan_out,activation="tanh")(X0)

        if mix:
            X1_tr = tf.transpose(X1, [0, 2, 1])
        else:
            X1_tr=X1

        Y = tf.reshape(X1_tr, [-1, nb_feature // fan_in * fan_out])

        return Y



""" Light Auto Encoder """
class LAE_simulator:

    def __init__(self, input_dim, struct_enc, reduce_dim, draw=False, only_enc=False, struct_dec=None, mix=True):
        self.initial_nb_feature=input_dim
        self.struct_enc=struct_enc
        self.compression_dim=reduce_dim
        self.draw=draw
        self.only_enc=only_enc
        self.struct_dec=struct_dec
        self.mix=mix

        self.verbose=False

        if struct_dec is None:
            struct_dec=[(b,a) for (a,b) in  struct_enc[::-1]]

        if only_enc:
            nb_layer = len(struct_enc) + 1
        else:
            nb_layer=len(struct_enc)+2+len(struct_dec)

        if self.draw:
            fig,ax=plt.subplots(figsize=(5*nb_layer,10))
        else:
            ax=None

        total_param=0
        total_param_if_full=0

        successive_feat_dim=[input_dim]

        nb_feat_in=input_dim
        depth=-1

        for (fan_in,fan_out) in struct_enc:
            depth+=1
            successive_feat_dim.append(str((fan_in,fan_out))+"->")
            nb_feat_out,nb_param=self.block_simulator(nb_feat_in,fan_in,fan_out,self.mix,depth,ax)
            successive_feat_dim.append(nb_feat_out)
            total_param+=nb_param
            total_param_if_full+=nb_feat_in*nb_feat_out
            nb_feat_in=nb_feat_out

        nb_fea_after_sparse=nb_feat_in
        #on rejoint la compression_dim
        depth += 1
        nb_feat_out, nb_param = self.block_simulator(nb_fea_after_sparse, nb_fea_after_sparse, self.compression_dim, False, depth, ax)
        successive_feat_dim.append(nb_feat_out)
        total_param += nb_param
        total_param_if_full += nb_feat_in * nb_feat_out

        if only_enc:
            print("successive_feat_dim of encoder only:",successive_feat_dim)
            return

        #on quite la compression_dim
        depth += 1
        nb_feat_out, nb_param = self.block_simulator(self.compression_dim, self.compression_dim, nb_fea_after_sparse, False, depth, ax)
        successive_feat_dim.append(nb_feat_out)
        total_param += nb_param
        total_param_if_full += nb_feat_in * nb_feat_out


        nb_feat_in=nb_fea_after_sparse

        for (fan_in,fan_out) in struct_dec:
            depth+=1
            successive_feat_dim.append(str((fan_in,fan_out))+"->")
            nb_feat_out,nb_param=self.block_simulator(nb_feat_in,fan_in,fan_out,self.mix,depth,ax)
            successive_feat_dim.append(nb_feat_out)

            total_param+=nb_param
            total_param_if_full+=nb_feat_in*nb_feat_out
            nb_feat_in=nb_feat_out

        print("successive_feat_dim of the compo:",successive_feat_dim)

        print("ratio total_param/total_param_if_full:",total_param/total_param_if_full)



    def block_simulator(self,nb_feature, fan_in, fan_out , mix, depth,ax):

        if nb_feature % fan_in != 0:
            raise Exception(f"lors de la couche {depth}, le nombre de feature: {nb_feature} n'est pas divisible par fan_in: {fan_in} ")

        X = tf.range(nb_feature)[tf.newaxis, :]

        X0 = tf.reshape(X, [-1, nb_feature // fan_in, fan_in])

        # X1=tf.keras.layers.Dense(fact,"tanh")(X0)
        X1 = tf.range(nb_feature // fan_in * fan_out)
        X1 = tf.reshape(X1, [-1, nb_feature // fan_in, fan_out])  # X1=tf.keras.layers.Dense(2)(X0)

        if mix:
            X1_tr = tf.transpose(X1, [0, 2, 1])
        else:
            X1_tr=X1

        Y = tf.reshape(X1_tr, [-1, nb_feature // fan_in * fan_out])

        if self.verbose:
            print("X0", X0.shape,"\n" ,X0.numpy())
            print("X1", X1.shape, "\n", X1.numpy())
            print("Y", Y.shape, "\n", Y.numpy())


        pos_X=tf.linspace(0, 1, X.shape[1])
        pos_Y=tf.linspace(0, 1, Y.shape[1])

        if self.draw:
            deb=1/2*depth
            end=1/2*(depth+1)

            for i in X[0, :]:
                ax.scatter(deb, pos_X[i])

            for j in Y[0, :]:
                ax.scatter(end, pos_Y[j])

            for a in range(X0.shape[1]):
                for i in X0[0, a, :]:
                    for j in X1[0, a, :]:
                        ax.plot([deb,end], [ pos_X[X[0, i]], pos_Y[Y[0, j]]])


        nb_param=fan_in*fan_out*X1.shape[1]
        return Y.shape[1],nb_param



def test_LFAE_simulator():
    LAE_simulator(input_dim=27, struct_enc=[(3, 2)], reduce_dim=4, only_enc=False, draw=True)
    LAE_simulator(input_dim=27, struct_enc=[(3, 2)], reduce_dim=4, only_enc=False, draw=False)
    LAE_simulator(input_dim=27, struct_enc=[(3, 2), (3, 2)], reduce_dim=4, only_enc=True, draw=True)

    plt.show()

    def test_LAE_fully_connected():
        input_dim = 10
        reduce_dim = 5
        struct_enc = [(10, 10)]
        model = LAE(input_dim, struct_enc, reduce_dim)
        model.encoder.summary()

        LAE_simulator(input_dim, struct_enc, reduce_dim, draw=True, only_enc=True)
        plt.show()

    def test_LAE_big():
        input_dim = 1000
        reduce_dim = 30
        struct_enc = [(100, 40), (50, 10), (20, 10)]

        LAE_simulator(input_dim, struct_enc, reduce_dim, draw=False)

    def test_assertion_error():
        input_dim = 1000
        reduce_dim = 30
        # struct_enc = [(100, 40), (50, 10), (20, 10)]
        struc_enc = [(10, 5), (10, 5), (10, 5), (5, 3), (5, 3)]

        LAE_simulator(input_dim, struc_enc, reduce_dim, only_enc=False)

    def test_LAE():
        input_dim = 27
        struct_enc = [(3, 2), (3, 2)]
        reduce_dim = 4
        model = LAE(input_dim, struct_enc, reduce_dim)

        batch_size = 5
        X = tf.ones([batch_size, input_dim])

        Y = model.encoder(X)
        Z = model.decoder(Y)
        Z_bis = model.compo(X)

        print("Y.shape", Y.shape)
        print("Z.shape", Z.shape)
        print("Z_bis.shape", Z_bis.shape)

        print(tf.reduce_sum(tf.abs(Z - Z_bis)))



def test_LAE_fully_connected():
    input_dim = 10
    reduce_dim=5
    struct_enc = [(10, 10)]
    model = LAE(input_dim, struct_enc, reduce_dim)
    model.encoder.summary()

    LAE_simulator(input_dim,struct_enc,reduce_dim,draw=True,only_enc=True)
    plt.show()



def test_LAE_big():
    input_dim = 1000
    reduce_dim=30
    struct_enc=[(100,40),(50,10),(20,10)]

    LAE_simulator(input_dim, struct_enc, reduce_dim,draw=False)

def test_assertion_error():
    input_dim = 1000
    reduce_dim = 30
    #struct_enc = [(100, 40), (50, 10), (20, 10)]
    struc_enc = [(10, 5), (10, 5), (10, 5), (5, 3), (5, 3),(5, 3)]

    LAE_simulator(input_dim, struc_enc, reduce_dim,only_enc=False)


def test_LAE():
    input_dim=27
    struct_enc=[(3,2),(3,2)]
    reduce_dim=4
    model=LAE(input_dim,struct_enc,reduce_dim)

    batch_size=5
    X=tf.ones([batch_size,input_dim])

    Y=model.encoder(X)
    Z=model.decoder(Y)
    Z_bis=model.compo(X)

    print("Y.shape",Y.shape)
    print("Z.shape",Z.shape)
    print("Z_bis.shape",Z_bis.shape)

    print(tf.reduce_sum(tf.abs(Z-Z_bis)))




if __name__=="__main__":
    #test_block_simulator()
    #test_LFAE_simulator()
    #test_LAE()
    #test_LAE_fully_connected()
    #test_LAE_big()
    test_assertion_error()
