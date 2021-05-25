import tensorflow as tf
import numpy as np
import Hamilton.autoencoder.data_maker as dat
import matplotlib.pyplot as plt
import Hamilton.autoencoder.autoencoder as models


def perdiodic_loss(y_true,y_pred,domain_size):
    diff=y_true-y_pred
    three_shifts=tf.abs(tf.stack([diff,diff-domain_size,diff+domain_size],axis=2))
    mini=tf.reduce_min(three_shifts,axis=2)
    res=tf.reduce_mean(mini)
    return res





class Circle_maker:

    def __init__(self,domain_size):
        self.domain_size=domain_size
        #ce paramètre n'a pas beaucoup d'importance, car il est multiplier par les weights de la première couche
        self.circle_radius=0.2


    def augmentation(self,X,):
        X_cos=tf.cos(X*2*np.pi/self.domain_size)*self.circle_radius
        X_sin=tf.sin(X*2*np.pi/self.domain_size)*self.circle_radius
        #pour un réseau dense, l'ordre des features n'a pas d'importance.
        #pour des réseaux structurés, il faudrait alterner les sinus et cosinus
        return tf.concat([X_cos,X_sin],axis=1)

    def unaugmentation(self, X):
        nb_fea=X.shape[1]//2
        X_cos=X[:,:nb_fea]
        X_sin=X[:,nb_fea:]
        angle=tf.atan2(X_sin,X_cos) #y,x
        return angle/(2*np.pi)*self.domain_size


def test_unaugmentation():
    DIM_INPUT = 16
    domain_size = 0.2

    curve = dat.Data_maker_curve_periodic(DIM_INPUT, domain_size,periodic=True)
    data = curve.make_sorted(300)

    augmenter=Circle_maker(domain_size)

    data_aug = augmenter.augmentation(data )
    data_unaug=augmenter.unaugmentation(data_aug)

    print("simple loss",tf.reduce_mean(tf.abs(data-data_unaug)))
    print("periodic loss",perdiodic_loss(data,data_unaug,domain_size))

    dat.present_data_margin(data)
    dat.present_data_margin(data_unaug)

    plt.show()






def test_augmentation_circular():
    DIM_INPUT = 16
    domain_size = 0.2

    curve = dat.Data_maker_curve_periodic(DIM_INPUT, domain_size,True)
    data = curve.make_sorted(300)

    circle_maker = Circle_maker(domain_size)
    data_aug = circle_maker.augmentation(data)

    print(data.shape)
    print(data_aug.shape)

    dat.present_data_margin(data_aug)
    dat.present_data(data_aug)

    plt.show()



def test_autoencoder():
    autoencoder = models.AutoEncoder(input_dim=100,reduced_dim=10)
    autoencoder.encoder.summary()




def test_periodic_loss():
    y_true=tf.constant([[0.9,0.9,1.1,1.1],[0.9,0.9,1.1,1.1]])
    y_pred=tf.constant([[0.9,1.1,0.9,1.1],[0.9,1.1,0.9,1.1]])

    diff=y_true-y_pred
    domain_size=1
    three_shifts = tf.abs(tf.stack([diff, diff - domain_size, diff + domain_size], axis=2))
    print("three_shifts",three_shifts)
    mini = tf.reduce_min(three_shifts, axis=2)
    res = tf.reduce_mean(mini)
    print("mini",mini)
    print("res",res)


if __name__=="__main__":
    #test_autoencoder()
    test_unaugmentation(False)
    #test_unaugmentation_circular()



