dim_1 = False
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from GCN.frontier_generator import Frontier

if dim_1:
    Conv = tf.keras.layers.Conv1D
    UpSampling = tf.keras.layers.UpSampling1D
    SpatialDropout = tf.keras.layers.SpatialDropout1D
else:
    Conv = tf.keras.layers.Conv2D
    UpSampling = tf.keras.layers.UpSampling2D
    SpatialDropout = tf.keras.layers.SpatialDropout2D

class VNet:

    def __init__(self,
                 input_shape,
                 levels=3,
                 depth=2,
                 kernel_size=3,
                 up_kernel_size=3,
                 activation="relu",
                 batch_norm=True,
                 dropout_rate=0,
                 ):

        self.input_shape = input_shape
        self.levels = levels
        self.depth = depth
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        if dim_1:
            assert len(input_shape) == 2
        else:
            assert len(input_shape) == 3

        inputs = tf.keras.Input(shape=input_shape)
        logits = self.body(inputs)
        #todo ou bien Dense
        outputs = Conv(1, 1, activation="sigmoid")(logits)

        self.keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)


    def doubleConv(self, Y, depth):
        Y = Conv(depth, self.kernel_size, activation=self.activation, padding='same')(Y)
        Y = Conv(depth, self.kernel_size, activation=self.activation, padding='same')(Y)
        if self.batch_norm:
            Y = tf.keras.layers.BatchNormalization()(Y)
        if self.dropout_rate > 0:
            Y = SpatialDropout(self.dropout_rate)(Y)
        return Y

    def makeDown(self, Y, depth):
        down = Conv(depth, 2, strides=2, padding="same")(Y)
        return down

    def makeUp(self, Y, depth):
        Y = UpSampling()(Y)
        up = Conv(depth, self.up_kernel_size, activation=self.activation, padding="same",
                  kernel_initializer=tf.keras.initializers.Constant(value=1 / self.up_kernel_size / depth))(Y)
        return up

    def body(self, inputs):
        left = dict()
        left[0] = self.doubleConv(inputs, self.depth)
        print("left[0].shape=", left[0].shape)
        for i in range(1, self.levels):
            down = self.makeDown(left[i - 1], self.depth * 2 ** i)
            conv = self.doubleConv(down, self.depth * 2 ** i)
            left[i] = tf.keras.layers.Add()([down, conv])
            if i < self.levels - 1:
                print(f"left[{i}].shape=", left[i].shape)

        central = left[self.levels - 1]
        print(f"central.shape=", central.shape)

        right = central
        for i in range(self.levels - 2, -1, -1):
            up = self.makeUp(right, self.depth * 2 ** i)
            add = tf.keras.layers.Add()([left[i], up])
            conv = self.doubleConv(add, self.depth * 2 ** i)
            right = tf.keras.layers.Add()([up, conv])
            print(f"right[{i}].shape=", right.shape)

        return right



def test_VNet():
    input_shape=(20,20,1)
    X=tf.zeros((3,)+input_shape)
    vNet=VNet(input_shape,2)
    Y=vNet.keras_model(X)
    print(Y.shape)


@tf.function
def dice_loss(y_true, y_pred,smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1-(2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def test_dice():
    y_true = np.zeros([7,10, 10, 1], dtype=np.float32)
    y_true[:,:6] = 1

    y_pred = np.zeros([7,10, 10, 1], dtype=np.float32)
    y_pred[:,5:] = 1

    print("some error",dice_loss(y_true,y_pred,smooth=0))

    print("no error",dice_loss(y_true, y_true, smooth=0))

    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(y_true[0,:, :, 0])
    ax0.set_title("y_true")
    ax1.imshow(y_pred[0,:, :, 0])
    ax1.set_title("y_pred")

    plt.show()







class Agent_frontier:

    def __init__(self,frontier,image_size=(100,100,1),batch_size=256):


        self.show_plots_after_train=True
        self.batch_size=batch_size

        assert len(image_size)==3
        self.image_size=image_size
        a=np.linspace(0,1,self.image_size[0],dtype=np.float32)
        b=np.linspace(0,1,self.image_size[1],dtype=np.float32)
        self.aa,self.bb=tf.meshgrid(a,b)
        self.aa_flat, self.bb_flat=tf.reshape(self.aa,[-1]),tf.reshape(self.bb,[-1])

        self.train_losses=[]
        self.val_losses=[]
        self.val_steps=[]

        self.frontier:Frontier=frontier
        self.vNet=VNet(self.image_size)

        """ Remarque: avec
        learning_rate=1e-3 (et plus petit)
        batch_size=20
        image_size=(40,40,1)
        l'apprentissage échoue de temps en temps """
        self.optimizer=tf.keras.optimizers.Adam(1e-2)


    @tf.function
    def train_step(self,X,Y):

        with tf.GradientTape() as tape:
            Y_pred=self.vNet.keras_model(X)
            loss=dice_loss(Y,Y_pred)

            variables=self.vNet.keras_model.trainable_variables
            gradients=tape.gradient(loss,variables)

        self.optimizer.apply_gradients(zip(gradients,variables))

        return loss



    def val_step(self,X,Y):
        Y_pred = self.vNet.keras_model(X)
        loss = dice_loss(Y,Y_pred)
        return loss


    def generate_data(self):
        X,Y=self.frontier.compute(self.aa_flat, self.bb_flat, self.batch_size)
        sh=(self.batch_size,)+self.image_size
        X,Y=tf.reshape(X,sh),tf.reshape(Y,sh)

        #todo: ça irait plus vite de faire une convolution 2D avec un padding same
        diff_x=tf.abs(Y[:,2:,:,:]-Y[:,:-2,:,:])
        ze_x=tf.zeros([self.batch_size,1,self.image_size[1],1])
        diff_x=tf.concat([ze_x,diff_x,ze_x],axis=1)

        diff_y=tf.abs(Y[:,:,2:,:]-Y[:,:,:-2,:])
        ze_y=tf.zeros([self.batch_size,self.image_size[0],1,1])
        diff_y=tf.concat([ze_y,diff_y,ze_y],axis=2)

        return X,tf.minimum(diff_x+diff_y,1.)


    def train(self,minutes=0.5):
        ti0=time.time()
        step=-1
        OK=True
        while OK:
            step+=1

            X,Y=self.generate_data()

            train_loss=self.train_step(X,Y).numpy()
            self.train_losses.append(train_loss)

            if step>0 and step%20==0:

                X, Y = self.generate_data()
                val_loss = self.val_step(X,Y).numpy()
                self.val_losses.append(val_loss)
                self.val_steps.append(step)

                if val_loss <= np.min(self.val_losses):
                    print(f"⤥:{val_loss*100:.4f}",end="")
                    self.best_val_loss = val_loss
                    self.best_step = step
                    self.vNet.keras_model.save("model_vNet.h5") #attention, si modèle complexe, il faut indiquer un répertoire

                else:
                    print(".",end="")

                OK = (time.time() - ti0)  < minutes* 60


        self.best_model = tf.keras.models.load_model("model_vNet.h5")



        if self.show_plots_after_train:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 4))
            ax1.scatter(self.best_step, self.best_val_loss)
            ax0.plot(np.arange(len(self.train_losses)), self.train_losses, label="train")
            ax1.plot(self.val_steps, self.val_losses, label="val")
            ax1.set_yscale('log')
            ax0.set_yscale('log')
            ax1.legend()
            ax0.legend()

            plt.show()



def test_agent_generate_data():
    agent=Agent_frontier(Frontier(kind=Frontier.kind_trivial))
    X,Y=agent.generate_data()
    print(X.shape,Y.shape)

    fig,axs=plt.subplots(5,2,figsize=(6,10))
    for i in range(5):
        axs[i,0].imshow(X[i,:,:,0],cmap="jet",interpolation=None)
        axs[i, 1].imshow(Y[i, :, :, 0],cmap="gray",interpolation=None)

    plt.show()


def test_agent_train():
    frontier=Frontier(kind=Frontier.kind_iles)
    agent = Agent_frontier(frontier,image_size=(40,40,1),batch_size=20)
    agent.train(0.5)

    X,Y=agent.generate_data()
    Y_pred=agent.best_model(X)
    fig, axs = plt.subplots(5, 3, figsize=(6, 10))
    for i in range(5):
        axs[i, 0].imshow(X[i, :, :, 0], cmap="jet", interpolation=None)
        axs[i, 1].imshow(Y_pred[i, :, :, 0], cmap="gray", interpolation=None)
        axs[i, 2].imshow(Y[i, :, :, 0], cmap="gray", interpolation=None)

        axs[i, 0].axis("off")
        axs[i, 1].axis("off")
        axs[i, 2].axis("off")

    axs[0,0].set_title("X")
    axs[0,1].set_title("Y_pred")
    axs[0,2].set_title("Y")


    plt.show()

def test_binary_crossentropy():
    Y=tf.ones([50])
    Y_pred=tf.ones([50])
    loss=tf.keras.losses.binary_crossentropy(Y,Y_pred)
    print(loss)


if __name__=="__main__":
    #test_VNet()
    #test_agent_generate_data()
    test_agent_train()
    #test_dice()





