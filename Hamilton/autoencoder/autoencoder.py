import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import Hamilton.autoencoder.agents as agents
import Hamilton.autoencoder.data_maker as dat
import Hamilton.autoencoder.utilities as util

np.set_printoptions(precision=3,suppress=True,linewidth=10000)
tfk = tf.keras
tfkl = tfk.layers
import numpy as np


class AutoEncoder:
    def __init__(self, input_dim,reduced_dim,coding_layer_no=4, coding_layer_rep=2 ):

        up_dims = []

        assert coding_layer_no >= 0

        # - Non linearity on hidden layers
        nonlinearity = "tanh"

        # - Weight initialisation
        initializer = tfk.initializers.Orthogonal()

        # - Computing number of units for each coding layer
        # - (every layer except in, out and bottom)
        coding_dims = []
        coding_dims += up_dims
        coding_dims.append(input_dim)
        step = (input_dim - reduced_dim - 1) / (coding_layer_no + 1)
        for k in range(coding_layer_no):
            coding_dims.append(int(input_dim - step * (k + 1)))
        print('coding layers dimensions: (without rep)', coding_dims)

        # - Build the encoder
        encoder_input = tfkl.Input([input_dim])
        encoder_layers = []
        for k in coding_dims:
            for _ in range(coding_layer_rep):
                encoder_layers.append(tfkl.Dense(k, activation=nonlinearity,
                                                 kernel_initializer=initializer)  # ,
                                      #    kernel_regularizer='l2')
                                      )

        for _ in range(coding_layer_rep):
            encoder_layers.append(tfkl.Dense(reduced_dim, activation=nonlinearity,
                                             kernel_initializer=initializer)  # ,
                                  #    kernel_regularizer='l2')
                                  )

        current = encoder_input
        for layer in encoder_layers:
            current = layer(current)
        self.encoder = tfk.Model(inputs=encoder_input, outputs=current,
                                 name='encoder')

        # - Build the decoder
        decoder_input = tfkl.Input([reduced_dim])
        decoder_layers = []
        for k in coding_dims[::-1]:
            for _ in range(coding_layer_rep):
                decoder_layers.append(tfkl.Dense(k, activation=nonlinearity,
                                                 kernel_initializer=initializer)  # ,
                                      #    kernel_regularizer='l2')
                                      )
        for _ in range(coding_layer_rep):
            decoder_layers.append(tfkl.Dense(input_dim, activation='linear',
                                             kernel_initializer=initializer)  # ,
                                  #    kernel_regularizer='l2')
                                  )

        current = decoder_input
        for layer in decoder_layers:
            current = layer(current)
        #
        # if self.domain_size is not None:
        #     current=tf.math.floormod(current,self.domain_size)-self.domain_size/2


        self.decoder = tfk.Model(inputs=decoder_input, outputs=current,
                                 name='decoder')

        # - Build the composition
        input_compo = tfkl.Input([input_dim])
        self.compo = tfk.Model(inputs=input_compo,
                               outputs=self.decoder(self.encoder(input_compo)),
                               name='compo'
                               )



def augmentation_circular(X,domain_size):
    #le facteur 0.2, c'est pour arriver dans une zone où l'activation tanh est relativement linéaire
    X_cos=tf.cos(X*2*np.pi/domain_size)*0.2
    X_sin=tf.sin(X*2*np.pi/domain_size)*0.2

    #pour un réseau dense, l'ordre des features n'a pas d'importance.
    #pour des réseaux structurés, il faudrait alterner les sinus et cosinus
    return tf.concat([X_cos,X_sin],axis=1)



def unaugmentation_circular(X,domain_size):
    nb_fea=X.shape[1]//2
    X_cos=X[:,:nb_fea]
    X_sin=X[:,nb_fea:]
    angle=tf.atan2(X_sin,X_cos) #y,x
    return angle/(2*np.pi)*domain_size


def test_unaugmentation_circular():
    DIM_INPUT = 16
    domain_size = 0.2

    curve = dat.Data_maker_curve_periodic(DIM_INPUT, domain_size,True)
    data = curve.make_sorted(300)
    data_aug = augmentation_circular(data, domain_size)
    data_unaug=unaugmentation_circular(data_aug,domain_size)

    print("simple loss",tf.reduce_mean(tf.abs(data-data_unaug)))
    print("periodic loss",util.perdiodic_loss(data,data_unaug,domain_size))

    dat.present_data_margin(data)
    dat.present_data_margin(data_unaug)

    plt.show()


def test_augmentation_circular():
    DIM_INPUT = 16
    domain_size = 0.2

    curve = dat.Data_maker_curve_periodic(DIM_INPUT, domain_size,True)
    data = curve.make_sorted(300)
    data_aug=augmentation_circular(data,domain_size)

    print(data.shape)
    print(data_aug.shape)

    dat.present_data_margin(data_aug)
    dat.present_data(data_aug)

    plt.show()



if __name__=="__main__":
    test_augmentation_circular()
    #test_unaugmentation_circular()
