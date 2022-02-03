import tensorflow as tf
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True, precision=2, linewidth=40000)
import matplotlib.pyplot as plt
import time


class DataGenerator_conv:

    def __init__(self, batch_size, seq_len,kernel_size,activation=None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.convo = tf.keras.layers.Conv1D(1, kernel_size, activation=activation, padding="same", trainable=False)

    def __call__(self):
        X = tf.random.uniform([self.batch_size, self.seq_len, 1], minval=-0.1, maxval=0.1)
        Y=self.convo(X)
        return X, Y


class DataGenerator_subsampling:

    def __init__(self, batch_size, seq_len,sampling_delta:int,averaging:bool):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.sampling_delta=sampling_delta
        self.averaging=averaging

    def __call__(self):
        X = tf.random.uniform([self.batch_size, self.seq_len, 1], minval=-0.1, maxval=0.1)
        if self.averaging:
            Y = tf.keras.layers.AveragePooling1D(self.sampling_delta)(X)
        else:
            Y = X[:, ::self.sampling_delta, :]

        Y = tf.keras.layers.UpSampling1D(self.sampling_delta)(Y)
        return X, Y


class DataGenerator_permut:

    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.permut=np.random.permutation(seq_len)

    def __call__(self):
        X = tf.random.uniform([self.batch_size, self.seq_len, 1], minval=-0.1, maxval=0.1)

        Y=tf.gather(X,self.permut,axis=1)

        return X, Y

def test_data_generator_permut():
    data_generator=DataGenerator_permut(1,3)
    x,y=data_generator()
    print(x)
    print(y)
#test_data_generator_permut()

class Preprocessor_sign:

    def __init__(self, seq_len, d_model, scale, nb_alternation):
        self.seq_len = seq_len
        self.d_model = d_model

        ind = tf.cast(tf.range(seq_len), tf.float32)

        alternation = []
        for i in range(1, nb_alternation + 1):
            alternation.append((-1) ** ((ind // i) % 2) / 2 * scale)

        self.alternation = tf.stack(alternation, axis=1)

        self.embed = tf.keras.layers.Dense(d_model,kernel_initializer=tf.keras.initializers.Orthogonal() , trainable=False)


    def __call__(self, x):
        batch_size = x.shape[0]
        pos_enc = tf.ones([batch_size, 1, 1]) * self.alternation[tf.newaxis, :, :]
        x = tf.concat([x, pos_enc], axis=2)
        x = self.embed(x)
        return x

class Preprocessor_minimal:
    def __init__(self, seq_len, d_model):
        self.seq_len = seq_len
        self.d_model = d_model
        self.embed = tf.keras.layers.Dense(d_model,kernel_initializer=tf.keras.initializers.Orthogonal() , trainable=False)

    def __call__(self, x):
        x = self.embed(x)
        return x


def test_preprocessor_sign():
    seq_len = 10
    d_model = 2
    scale = 0.2
    nb_alternation = 2
    x = tf.random.uniform([1, seq_len, d_model], -0.1, 0.1)
    prepros = Preprocessor_sign(seq_len, d_model, scale, nb_alternation)
    x = prepros(x)
    print(x.shape)


def transformer_encoder(inputs, head_size, num_heads, ff_dim):
    # Normalization and Attention
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(x, x)
    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer(seq_len,d_model,head_size,num_heads,ff_dim,num_transformer_blocks,dropout=0,):
    inputs = tf.keras.Input(shape=(seq_len, d_model))
    x = inputs

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim)

    outputs = tf.keras.layers.Conv1D(1, 1)(x)
    return tf.keras.Model(inputs, outputs)

class ModelConvo(tf.keras.layers.Layer):
    def __init__(self,d_model,nb_levels):
        super(ModelConvo,self).__init__()
        self.layers=[]
        for _ in range(nb_levels):
            self.layers.append(tf.keras.layers.Conv1D(d_model,2,padding="same"))

        self.final_layer=tf.keras.layers.Conv1D(1,1,padding="same")

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return self.final_layer(x)


def test_build_model():
    seq_len = 10
    d_model = 8
    head_size = 3
    num_heads = 4
    ff_dim = 5
    num_transformer_blocks = 3
    dropout = 0

    model_transformer = build_transformer(seq_len, d_model, head_size, num_heads, ff_dim, num_transformer_blocks, dropout)
    model_convo=ModelConvo(d_model,3)

    batch_size = 1
    x = tf.random.uniform([batch_size, seq_len, 1])
    preprocess = Preprocessor_sign(seq_len, d_model, 1, 2)

    pred_transformer = model_transformer(preprocess(x))
    pred_convo = model_convo(preprocess(x))

    print("pred_transformer.shape:", pred_transformer.shape)
    print("pred_convo",pred_convo.shape)


#test_build_model()

class MultiTrainer:
    def __init__(self, data_generators: dict, preprocessors: dict, model_builders: dict):

        self.trainers=[]
        self.caracteristics=[]
        for data_gene_name,data_generator in data_generators.items():
            for prepro_name,preprocessor in preprocessors.items():
                for model_name,model_builder in model_builders.items():
                    #attention, il faut créer un nouveau modèle pour chaque trainer
                    self.trainers.append(Trainer(data_generator,preprocessor, model_builder()))
                    self.caracteristics.append(pd.Series(
                        {
                         "data_gen":data_gene_name,
                         "prepro":prepro_name,
                         "model":model_name,
                         "error":float("inf")}
                    )
                    )

    def go(self,minutes_per_epoch, nb_epoch):
        self.result = {}
        self.summary={}

        for _ in range(nb_epoch):
            for trainer,caracteristic in zip(self.trainers,self.caracteristics):
                try:
                    print("Début entrainement de:\n"+str(caracteristic))
                    trainer.train(minutes_per_epoch)
                except KeyboardInterrupt:
                    pass
                error=trainer.eval()
                old_error=caracteristic["error"]
                if old_error>error:
                    caracteristic["error"]=error
                    print(f"amélioration. Ancienne erreur:{old_error}, nouvelle erreur:{error}")

        carac_sorted=sorted(self.caracteristics,key=lambda elem:elem["error"])
        print(pd.DataFrame(carac_sorted))


class Trainer:

    def __init__(self, data_generator, preprocessor, predictor):
        self.predictor = predictor
        self.data_generator = data_generator
        self.preprocessor = preprocessor

        self.train_losses = []

        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.batch_size = 64


    @tf.function
    def train_step(self, X, Y):

        with tf.GradientTape() as tape:
            predictions = self.predictor(self.preprocessor(X))
            loss = tf.reduce_mean((Y - predictions) ** 2)

        gradients = tape.gradient(loss, self.predictor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.predictor.trainable_variables))
        return loss


    def train(self, minutes_per_epoch):

        start = time.time()

        while time.time() - start < minutes_per_epoch * 60:
            X, Y = self.data_generator()
            loss = self.train_step(X, Y)
            self.train_losses.append(loss)

        # ind=tf.cast(tf.range(len(self.train_losses)),dtype=tf.float32)
        fig, axs = plt.subplots(1, 1, sharex="all")
        axs.plot(self.train_losses)
        axs.set_yscale("log")
        # axs[1].plot(ind,self.learning_rate_fn(ind))
        plt.show()


    def eval(self):
        X, Y = self.data_generator()
        pred = self.predictor(self.preprocessor(X))

        error_along_time = tf.reduce_mean(tf.abs(Y - pred), axis=1)
        max_along_time = tf.reduce_max(tf.abs(Y), axis=1)
        relative_error = tf.reduce_mean(error_along_time / max_along_time)

        ind = tf.range(X.shape[1])
        nb = 5
        fig, axs = plt.subplots(nb, 1)
        print(pred.shape)
        for i in range(nb):
            axs[i].plot(ind, Y[i, :, 0], label="true")
            axs[i].plot(ind, pred[i, :, 0], label="pred")
            # axs[i].plot(ind, X[i, :,0],".",label="input")

        axs[0].legend()
        plt.show()

        return relative_error.numpy()


def test_agent():
    seq_len = 60
    batch_size = 64
    d_model = 64

    data_generator = DataGenerator_conv(batch_size, seq_len,1)
    x, _ = data_generator()
    scale = tf.math.reduce_std(x)
    preprocess = Preprocessor_sign(seq_len, d_model, scale, 8)

    model = build_transformer(seq_len, d_model, head_size=64, num_heads=4, ff_dim=64, num_transformer_blocks=8)  # 256

    trainer = Trainer(data_generator, preprocess, model)
    try:
        trainer.train(0.1)
        trainer.eval()
    except KeyboardInterrupt:
        pass
    trainer.eval()


def test_multi_trainer():
    seq_len = 60
    batch_size = 64
    d_model = 64

    dataGen_conv=DataGenerator_conv(batch_size,seq_len,2)
    dataGen_sample=DataGenerator_subsampling(batch_size,seq_len,2,False)
    dataGen_permut=DataGenerator_permut(batch_size,seq_len)

    preprocess = Preprocessor_sign(seq_len, d_model, 1, 8)
    preprocess_mini=Preprocessor_minimal(seq_len,d_model)

    transformer_builder =lambda :build_transformer(seq_len, d_model, head_size=64, num_heads=4, ff_dim=64, num_transformer_blocks=8)  # 256
    model_convo_builder =lambda :ModelConvo(d_model,3)

    multi= MultiTrainer(
        {#"data_conv":dataGen_conv,"data_sample":dataGen_sample,
         "data_permut":dataGen_permut},
        {"prepro_sign":preprocess,"prepro_mini":preprocess_mini},
        {"model_transformer":transformer_builder,"model_convo":model_convo_builder})
    multi.go(0.2,2)


test_multi_trainer()



