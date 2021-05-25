from popup_lib.popup import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def weights_info_to_register(self) -> Dict[str, float]:
    res = {}
    for layer in self.keras_model.layers:
        for wei in layer.weights:
            # le layer.name est inclus dans le wei.name
            res[wei.name + "_mean"] = float(np.mean(wei.numpy()))
            res[wei.name + "_std"] = float(np.std(wei.numpy()))

    return res


# %%
family_trainers[0].history.metrics_values.keys()


# %%

def plot_weights(fm):
    fig, axs = plt.subplots(4, 1)
    for k in fm.history.metrics_values.keys():
        ax = None
        if k.endswith("mean") and "bias" in k:
            ax = axs[0]
            ax.set_title("mean of bias")

        elif k.endswith("std") and "bias" in k:
            ax = axs[1]
            ax.set_title("std of bias")

        elif k.endswith("mean") and "kernel" in k:
            ax = axs[2]
            ax.set_title("mean of kernel")

        elif k.endswith("std") and "kernel" in k:
            ax = axs[3]
            ax.set_title("std of kernel")

        if ax:
            ax.plot(fm.history.metrics_times[k], fm.history.metrics_values[k])

    fig.tight_layout()


plot_weights(family_trainers[0])


def present_fashion(X,Y):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i], cmap="gray")
        plt.xlabel(class_names[Y[i]])

    plt.show()


def test_fashion():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print("train_images.shape",train_images.shape)
    print("test_images.shape",test_images.shape)
    print("train_labels",train_labels)
    print("test_labels",test_labels)

    xy_dealer=XY_dealer()
    X,Y=xy_dealer.get_XY_train(25)
    present_fashion(X,Y)


class XY_dealer:

    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (X_train, Y_train), (X_val_test, Y_val_test) = fashion_mnist.load_data()
        X_train = X_train / 255.0
        X_val_test = X_val_test / 255.0

        nb_val=5000
        self.X_val=X_val_test[:nb_val]
        self.Y_val=Y_val_test[:nb_val]

        self.X_train=X_train
        self.Y_train=Y_train

        self.shuffle()


    def get_XY_val(self):
        perm=np.random.permutation(len(self.X_val))
        return self.X_val[perm[:400]],self.Y_val[perm[:400]]


    def shuffle(self):
        #print("new epoch")
        perm=np.random.permutation(len(self.X_train))

        self.X_train_shuffle=self.X_train[perm]
        self.Y_train_shuffle=self.Y_train[perm]
        self.batch_count=0

    def get_XY_train(self,batch_size):
        i=self.batch_count
        self.batch_count+=1

        if (i+1)*batch_size>=len(self.X_train):
            self.shuffle()

        X_batch=self.X_train_shuffle[i*batch_size:(i+1)*batch_size]
        Y_batch = self.Y_train_shuffle[i * batch_size:(i + 1) * batch_size]

        return X_batch,Y_batch



class Agent_convo(Abstract_Agent):

    def __init__(self,lr,batch_size,conv_struc=(10,20,10),dense_struc=(20,)):

        self.famparams = {"batch_size": batch_size, "lr": lr}
        self.conv_struc=conv_struc
        self.dense_struc=dense_struc

        self.model= self.make_model()
        self.xy_dealer = XY_dealer()
        self._count = -1

        self.make_optimizer()

    def get_famparams(self):
        return self.famparams

    def set_famparams(self, dico):
        self.famparams=dico
        self.make_optimizer()

    def make_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.famparams["lr"])

    def perturb_famparams(self,period_count):
        batch_size=int(self.famparams["batch_size"]*np.random.uniform(0.5,2))
        self.famparams["batch_size"]=np.clip(batch_size,16,1024)
        self.famparams["lr"]*=np.random.uniform(0.5,5)

    def make_model(self):
        inputs=tf.keras.layers.Input([28,28,1])

        current=inputs
        for nb_units in self.conv_struc:
            current=tf.keras.layers.Conv2D(nb_units,3)(current)

        current=tf.keras.layers.Flatten()(current)
        for nb_units in self.dense_struc:
            current=tf.keras.layers.Dense(nb_units,activation="softmax")(current)

        probas=tf.keras.layers.Dense(10,activation="softmax")(current)

        return tf.keras.Model(inputs=inputs,outputs=probas)


    def optimize_and_return_score(self) -> float:
        self._count+=1

        for _ in range(5):
            X,Y=self.xy_dealer.get_XY_train(self.famparams["batch_size"])

            with tf.GradientTape() as tape:
                Y_pred=self.model(X)
                loss=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(Y, Y_pred))

            gradients=tape.gradient(loss,self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))

        X, Y = self.xy_dealer.get_XY_val()
        Y_pred = self.model(X)
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(Y, Y_pred)
        return accuracy.numpy().mean()


    def set_weights(self, weights:List):
        self.model.set_weights(weights)

    def get_copy_of_weights(self) -> List:
        res=[]
        for tensor in self.model.get_weights():
            res.append(tensor) #get_weights tenvoie déjà des copies numpy
        return res

    def to_register_on_mutation(self) ->Dict[str, float]:
        res = {}
        for layer in self.model.layers:
            for wei in layer.weights:
                #le layer.name est inclus dans le wei.name
                res[wei.name+"_mean"] = float(np.mean(wei.numpy()))
                res[wei.name + "_std"] = float(np.std(wei.numpy()))

        return res


def test_model():
    agent=Agent_convo(1e-3,3)
    model=agent.make_model()
    X=np.zeros([3,28,28,1])
    res=model(X)
    print(res)
    for layer in model.layers:
        print(layer.name)
        for wei in layer.weights:
            print("\t"+wei.name,"shape:",wei.numpy().shape)

    print(agent.to_register_on_mutation())



def test_one():
    agent=Agent_convo(1e-4,256)
    family_trainer=Family_trainer([agent],period_for_each="10 steps")
    try:
        for _ in range(10):
            family_trainer.period()
    except KeyboardInterrupt:
        print("interuption manuelle")

    fig,(ax0,ax1)=plt.subplots(2,1,sharex='all')
    family_trainer.plot_metric("score",ax0)
    plt.show()


def main():
    def random_lr():
        return 10**np.random.uniform(-4,-2)

    popsize = 5
    agents_light=[Agent_convo(random_lr(),64,conv_struc=(10,20,10),dense_struc=(50,)) for _ in range(popsize)]
    agents_heavy=[Agent_convo(random_lr(),64,conv_struc=(32,64,32),dense_struc=(512,)) for _ in range(popsize)]

    family_trainers = [
        Family_trainer(agents_light,name="light", color="g", period_for_each="3 seconds"),
        Family_trainer(agents_heavy,name="heavy" ,color="r", period_for_each="3 seconds")
    ]
    try:
        for _ in range(10):
            for fm in family_trainers:
                fm.period()
    except KeyboardInterrupt:
        print("interuption manuelle")

    for fm in family_trainers:
        print(f"\n stats_of_best for {fm.name}:\n", fm.stats_of_best())

    fig,ax=plt.subplots()
    for fm in family_trainers:
        fm.plot_metric("score",ax)

    fig,ax=plt.subplots()
    for fm in family_trainers:
        fm.plot_metric("batch_size",ax)


    fig,ax=plt.subplots()
    for fm in family_trainers:
        fm.plot_metric("lr",ax)


    plt.show()

#
# def main_old():
#     agents=[]
#     def random_lr():
#         return 10**np.random.uniform(-4,-2)
#     for _ in range(5):
#         agents.append(Agent_convo(random_lr(),64))
#
#     family_trainer = Family_trainer(agents, period_duration=1, period_duration_unity="seconde")
#
#     try:
#         for _ in range(10):
#             family_trainer.period()
#     except KeyboardInterrupt:
#         print("interuption manuelle")
#
#     print("\n stats_of_best:\n", family_trainer.stats_of_best())
#
#     fig,ax=plt.subplots()
#     family_trainer.plot_metric("score",ax)
#
#     fig,ax=plt.subplots()
#     family_trainer.plot_metric("batch_size",ax)
#
#     fig,ax=plt.subplots()
#     family_trainer.plot_metric("lr",ax)
#
#     plt.show()

if __name__=="__main__":
    main()
#test_one()

