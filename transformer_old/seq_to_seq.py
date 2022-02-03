import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True,precision=2,linewidth=40000)
import matplotlib.pyplot as plt
import time


class DataGenerator:

    def __init__(self,batch_size,seq_len):
        self.batch_size=batch_size
        self.seq_len=seq_len

    def __call__(self):
        X=tf.random.uniform([self.batch_size,self.seq_len,1])
        Y=X[:,::2,:]
        Y=tf.keras.layers.UpSampling1D(2)(Y)
        return X,Y


class Preprocessor_sign:

    def __init__(self,seq_len,d_model,scale,nb_alternation):
        self.seq_len=seq_len
        self.d_model=d_model

        ind=tf.cast(tf.range(seq_len),tf.float32)

        alternation=[]
        for i in range(1,nb_alternation+1):
            alternation.append((-1)**( (ind//i) %2)/2*scale)

        self.alternation=tf.stack(alternation,axis=1)

        self.embed=tf.keras.layers.Dense(d_model,trainable=False)

    def __call__(self,x):
        batch_size=x.shape[0]
        pos_enc=tf.ones([batch_size,1,1])*self.alternation[tf.newaxis,:,:]
        x=tf.concat([x,pos_enc],axis=2)
        x=self.embed(x)
        return x



def test_preprocessor_sign():
    batch_size=1
    seq_len=10
    d_model=2

    x,_=DataGenerator(batch_size,seq_len)()

    scale=tf.math.reduce_std(x)
    print("scale",scale)

    prepros=Preprocessor_sign(seq_len,d_model,scale,4)# divisé la scale par le nombre d'alternation ?
    print(prepros.alternation)

    x=prepros(x)
    print(x)


#test_preprocessor_sign()


#
# def get_angles(pos, i, d_model):
#   angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
#   return pos * angle_rates
#
# def positional_encoding(position, d_model):
#   angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                           np.arange(d_model)[np.newaxis, :],
#                           d_model)
#
#   # apply sin to even indices in the array; 2i
#   angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#
#   # apply cos to odd indices in the array; 2i+1
#   angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#
#   pos_encoding = angle_rads[np.newaxis, ...]
#
#   return tf.cast(pos_encoding, dtype=tf.float32)


def test_data_generator():
    data_generator=DataGenerator(1,10)
    X,Y=data_generator()
    print(X)
    print(Y)



def scaled_dot_product_attention(v, k, q, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading (=last) dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      v: value shape == (..., seq_len_v, depth_v)
      k: key shape == (..., seq_len_k, depth)
      q: query shape == (..., seq_len_q, depth)

      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wv = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wq = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)

        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)

        """
        la scaled_dot_product_attention est appliquée sur les différentes tête.
        Cette fonction agit sur les deux dernière dimensions des tenseurs, qui sont les (seq_len_x, depth)
        """
        scaled_attention, attention_weights = scaled_dot_product_attention(v, k, q, mask)
        # scaled_attention.shape : (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape: (batch_size, num_heads, seq_len_v, seq_len_q)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights



def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class TransformatorLayer(tf.keras.layers.Layer):
    def __init__(self,  d_model, num_heads, dff, rate=0.1):
        super(TransformatorLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def __call__(self, x, training):

        attn1, attn_weights_block1 = self.mha1(x, x, x, None)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1


class Transformator(tf.keras.layers.Layer):

    def __init__(self, seq_len,num_layers, d_model, num_heads, dff, prepros,rate=0.1):
        super(Transformator, self).__init__()

        self.seq_len=seq_len
        self.d_model = d_model
        self.num_layers = num_layers

        self.embed=tf.keras.layers.Dense(d_model)
        self.unembed=tf.keras.layers.Dense(1)

        self.dec_layers = [TransformatorLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.prepros=prepros

    def __call__(self, x, training):

        attention_weights = {}

        x=self.prepros(x)
        #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attention_weight = self.dec_layers[i](x, training)

            attention_weights[f'decoder_layer{i + 1}_block1'] = attention_weight

        x=self.unembed(x)

        return x, attention_weights



class Trainer:

    def __init__(self,predictor,data_generator):
        self.predictor=predictor
        self.data_generator=data_generator

        self.train_losses=[]

        self.predictor=predictor
        self.optimizer=tf.keras.optimizers.Adam(1e-4)

        self.batch_size=64
        self.recurence_len=2


    #@tf.function
    def train_step(self, X, Y):

        with tf.GradientTape() as tape:
            predictions, _ = self.predictor(X, True)
            loss = tf.reduce_mean((Y - predictions) ** 2)

        gradients = tape.gradient(loss, self.predictor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.predictor.trainable_variables))
        return loss


    def main_loop(self,minutes_per_epoch,nb_epoch):

        for epoch in range(nb_epoch):
            start = time.time()

            while time.time()-start<minutes_per_epoch*60:
                X,Y= self.data_generator()
                loss=self.train_step(X,Y)
                self.train_losses.append(loss)

            print(f'duration of the {epoch}-th epoch: {time.time() - start:.2f} secs\n')
            plt.plot(self.train_losses)
            plt.yscale("log")
            plt.show()


    def eval(self):
        X,Y = self.data_generator()
        pred,_=self.predictor(X,False)
        ind=tf.range(X.shape[1])
        nb=5
        fig,axs=plt.subplots(nb,1)
        print(pred.shape)
        for i in range(nb):
            axs[i].plot(ind,Y[i,:,0],label="true")
            axs[i].plot(ind, pred[i, :,0],label="pred")

        axs[0].legend()
        plt.show()

def test_agent():
    seq_len=60
    batch_size=64
    d_model=64

    data_generator=DataGenerator(batch_size,seq_len)
    x,_=data_generator()
    scale=tf.math.reduce_std(x)

    prepros=Preprocessor_sign(seq_len,d_model,scale,4)

    predictor=Transformator(seq_len,num_layers=10 ,d_model=d_model, num_heads=4, dff=64,prepros=prepros, rate=0)

    trainer=Trainer(predictor,data_generator)
    try:
        trainer.main_loop(0.2,2)
    except KeyboardInterrupt:
        pass
    trainer.eval()

test_agent()








