import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

import Hamilton.autoencoder.data_maker as dat
import Hamilton.autoencoder.utilities as util
import Hamilton.autoencoder.autoencoder as models
import Hamilton.autoencoder.autoencoder_light as models_light



class Agent:

    kind_augmentation_circular="kind_augmentation_circular"

    kind_periodic_loss="kind_periodic_loss"
    kind_normal= "kind_normal"


    def __init__(self,model_maker, reduced_dim, data_maker,domain_size,kind:str,gradient_tape_watch_unaugmentation=False,batch_size = 256):

        self.model_maker=model_maker
        self.data_maker = data_maker
        self.domain_size=domain_size
        self.kind=kind
        self.gradient_tape_watch_unaugmentation=gradient_tape_watch_unaugmentation


        if self.kind!="kind_normal":
            assert self.domain_size is not None, "il faut préciser la taille du domaine (sauf pour la kind normal)"


        self.show_plots_after_train = True
        self.batch_size = batch_size

        self.train_losses = []
        self.val_losses = []
        self.val_steps = []

        self.optimizer = tf.keras.optimizers.Adam(1e-3)

        self.X_val_origin = self.data_maker.make(1000)
        self.dim_origin=self.X_val_origin.shape[1]


        if self.kind==Agent.kind_augmentation_circular :
            self.augmenter=util.Circle_maker(self.domain_size)
            self.X_val_augmented = self.augmenter.augmentation(self.X_val_origin)
            input_dim = 2 * self.dim_origin
        else:
            self.augmenter=None
            input_dim = self.dim_origin

        self.autoencoder =self.model_maker(input_dim,reduced_dim)


    def eval_model(self,x):
        # if self.kind == Agent.kind_augmentation_circular_twice:
        #     z0 = self.autoencoder.compo(x[:, self.dim_origin:])
        #     z1 = self.autoencoder.compo(x[:, :self.dim_origin])
        #     z = tf.concat([z0, z1], axis=1)
        return  self.autoencoder.compo(x)


    @tf.function
    def train_step(self, x):

        with tf.GradientTape() as tape:
            #attention Guillaume, tu avais mis un tape.watch(x) ici: inutile

            z=self.eval_model(x)

            if  self.kind==Agent.kind_periodic_loss:
                #au cas où le z doit vraiment loin du domaine
                z=tf.math.floormod(z,self.domain_size)-self.domain_size/2
                loss = util.perdiodic_loss(x,z,self.domain_size)
            else:
                loss= tf.reduce_mean(tf.abs(z-x))

        dtheta_loss = tape.gradient(loss, self.autoencoder.compo.trainable_variables)
        self.optimizer.apply_gradients(zip(dtheta_loss, self.autoencoder.compo.trainable_variables))
        return loss



    @tf.function
    def train_step_looking_augmentation(self, x_origin,x_aug):
        with tf.GradientTape() as tape:
            z = self.eval_model(x_aug)
            x_pred=self.augmenter.unaugmentation(z)
            loss = util.perdiodic_loss(x_pred, x_origin, self.domain_size)

        gradients = tape.gradient(loss, self.autoencoder.compo.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.autoencoder.compo.trainable_variables))

        return loss

    @tf.function
    def val_step_looking_unaugmentation(self, x_origin, x_aug, domain_size):
        z = self.eval_model(x_aug)
        x_pred = self.augmenter.unaugmentation(z)
        loss = util.perdiodic_loss(x_pred, x_origin, domain_size)
        return loss


    @tf.function
    def val_step(self, x,domain_size):
        z = self.eval_model(x)

        if  self.kind==Agent.kind_periodic_loss:
            z=tf.math.floormod(z, self.domain_size) - self.domain_size / 2
            loss = util.perdiodic_loss(x, z, domain_size)
        else:
            loss = tf.reduce_mean(tf.abs(z - x))

        return loss


    def predict(self,X_origin):
        #lancée après l'apprentissage, elle utilisera donc les meilleurs poids (pour le jeu de validation)
        if self.augmenter is not None:
            X_aug=self.augmenter.augmentation(X_origin)
            Y_aug=self.eval_model(X_aug)
            X_pred=self.augmenter.unaugmentation(Y_aug)
        else:
            X_pred=self.eval_model(X_origin)
            if self.kind==Agent.kind_periodic_loss:
                X_pred=tf.math.floormod(X_pred, self.domain_size) - self.domain_size / 2

        final_loss=util.perdiodic_loss(X_origin,X_pred,self.domain_size)

        return X_pred,final_loss


    def train(self, minutes=0.5):
        ti0 = time.time()
        step = -1
        OK = True
        while OK:
            step += 1

            X_origin = self.data_maker.make(self.batch_size)

            if self.augmenter is not None:
                X_augmented=self.augmenter.augmentation(X_origin)
                if self.gradient_tape_watch_unaugmentation:
                    train_loss = self.train_step_looking_augmentation(X_origin, X_augmented).numpy()
                else:
                    train_loss = self.train_step(X_augmented).numpy()
            else:
                train_loss=self.train_step(X_origin)


            self.train_losses.append(train_loss)

            if step > 0 and step % 20 == 0:
                if self.augmenter is not None:
                    if self.gradient_tape_watch_unaugmentation:
                        val_loss = self.val_step_looking_unaugmentation(self.X_val_origin, self.X_val_augmented,self.domain_size).numpy()
                    else:
                        val_loss = self.val_step(self.X_val_augmented, self.domain_size).numpy()
                else:
                    val_loss = self.val_step(self.X_val_origin,self.domain_size).numpy()


                self.val_losses.append(val_loss)
                self.val_steps.append(step)

                if val_loss <= np.min(self.val_losses):
                    print(f"⤥:{val_loss * 100:.4f}", end="")
                    self.best_val_loss = val_loss
                    self.best_step = step
                    self.best_weights=self.autoencoder.compo.get_weights()

                else:
                    print(".", end="")

                OK = (time.time() - ti0) < minutes * 60

        self.autoencoder.compo.set_weights(self.best_weights)


        if self.show_plots_after_train:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 4))
            ax1.scatter(self.best_step, self.best_val_loss)
            ax0.plot(np.arange(len(self.train_losses)), self.train_losses, label="train")
            ax1.plot(self.val_steps, self.val_losses, label="val")
            ax1.set_yscale('log')
            ax0.set_yscale('log')
            ax1.legend()
            ax0.legend()



def test_agent():

    #pour débugguer plus facilement
    tf.executing_eagerly()

    DIM_INPUT = 16
    DIM_COMPRESSION = 4
    #la taille du domaine est un paramètre important pour l'apprentissage !
    domain_size=0.2

    #data_maker=dat.Data_maker_curve_periodic(DIM_INPUT,domain_size,periodic=False)
    data_maker=dat.Data_maker_curve_periodic(DIM_INPUT,domain_size,periodic=False)

    def model_maker(input_dim,reduced_dim):
        return models.AutoEncoder(input_dim,reduced_dim)

    agent=Agent(model_maker,DIM_COMPRESSION, data_maker, domain_size=domain_size, kind=Agent.kind_normal,gradient_tape_watch_unaugmentation=True)
    agent.train(0.1)

    data = data_maker.make_sorted(1000)
    prediction,final_loss = agent.predict(data)

    print("\nla loss finale est de:",final_loss)

    #compressed = agent.autoencoder.encoder(data)
    #print(data.shape, compressed.shape, prediction.shape)

    dat.present_data_margin(data, prediction)
    dat.present_data(data, prediction)

    plt.show()



if __name__=="__main__":
    #test_periodic_loss()
    test_agent()

