import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True,precision=2,linewidth=40000)
import matplotlib.pyplot as plt
import time


def random_orthodonal_matrices(batch_size,recurence_len):
    liste=[]
    while len(liste)<batch_size:
        transitions = tf.random.uniform([recurence_len, recurence_len])
        U, _, _ = np.linalg.svd(transitions)
        liste.append(U)
    return tf.stack(liste)

def data_generation_complex(batch_size, recurence_len, seq_len):

    assert recurence_len<seq_len
    transitions=random_orthodonal_matrices(batch_size,recurence_len)
    init=tf.random.uniform([batch_size,recurence_len],-1,1)

    res=[init]
    while len(res)*recurence_len<seq_len:
        last=res[-1] #(batch,recurence_len)
        #sum_k transitions[i,j,k]*last[i,k]
        current= tf.reduce_sum(transitions*last[:,tf.newaxis,:],axis=2)
        res.append(current)

    res=tf.stack(res) #(seq_len/3,batch_size,recurence_len)
    res=tf.transpose(res,[1,0,2])
    res=tf.reshape(res,[batch_size,-1])

    return res[:,:seq_len]

class Data_generation_simple:

    def __init__(self,batch_size,seq_len):
        self.batch_size=batch_size
        self.seq_len=seq_len

    def __call__(self):
        init=tf.random.uniform([self.batch_size],-1,1)
        pente=tf.random.uniform([self.batch_size],-1,1)
        ind=tf.range(self.seq_len,dtype=tf.float32)
        return init[:,tf.newaxis]+pente[:,tf.newaxis]*ind[tf.newaxis,:]

def test_data_generation_complex():
    batch_size, recurence_len, seq_len = 5,2,61
    #res=data_generation_complex(batch_size, recurence_len, seq_len)
    generator=Data_generation_simple(batch_size,seq_len)
    res=generator()

    print("shape (batch_size,seq_len)",res.shape)

    fig,axs=plt.subplots(batch_size,1)
    ind=tf.range(seq_len)
    for i in range(batch_size):
        axs[i].plot(ind,res[i])

    plt.show()




def very_simple_embeding(x, d_model):
    # x.shape=(batch_size,seq_len)
    one = tf.ones([1, 1, d_model])
    return x[:, :, tf.newaxis] * one


def very_simple_unembeding(x):
    return tf.reduce_mean(x, axis=2)


def test_embed():
    temp_x = tf.constant([[1., 2, 3], [2, 4, 6]])
    temp_x_emb = very_simple_embeding(temp_x, 4)
    print(temp_x_emb)
    temp_x_recov = very_simple_unembeding(temp_x_emb)
    print(temp_x_recov)


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


def test_multihead_attention():
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print("out,attn:", out.shape, attn.shape)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class PredictorLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(PredictorLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def __call__(self, x, training, look_ahead_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1


class PredictorViaAttention(tf.keras.layers.Layer):

    def __init__(self, seq_len,num_layers, d_model, num_heads, dff, rate=0.1):
        super(PredictorViaAttention, self).__init__()

        self.seq_len=seq_len

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [PredictorLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.look_ahead_mask=self.create_look_ahead_mask(self.seq_len)[tf.newaxis,tf.newaxis,:,:]

    def create_look_ahead_mask(self,size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def __call__(self, x, training):
        # x.shape = (batch_size,seq_len)

        x = very_simple_embeding(x, self.d_model)
        # x.shape = (batch_size,seq_len,d_model)

        attention_weights = {}

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x, training, self.look_ahead_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1

        # x.shape == (batch_size, target_seq_len, d_model)

        x = very_simple_unembeding(x)

        return x, attention_weights



class AgentGRU:

    def __init__(self):

        super(AgentGRU,self).__init__()
        nb_layers=4
        d_model=64

        self.layers=[]

        for _ in range(nb_layers):
            self.layers.append(tf.keras.layers.GRU(d_model,return_sequences=True))

    def __call__(self,x):
        y=x
        for layer in self.layers:
            y=layer(y)

        return y



class Trainer:

    def __init__(self,predictor,data_generator):
        self.predictor=predictor
        self.data_generator=data_generator

        self.nb_epoch=1
        self.minutes_per_epoch=0.5
        self.train_losses=[]

        self.predictor=predictor
        self.optimizer=tf.keras.optimizers.Adam(1e-3)

        self.batch_size=64
        self.recurence_len=2
        self.seq_len=60


    #@tf.function
    def train_step(self, X):
        X_inp = X[:, :-1]
        X_real = X[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.predictor(X_inp,False)
            loss = tf.reduce_mean((X_real-predictions)**2)

        gradients = tape.gradient(loss, self.predictor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.predictor.trainable_variables))
        return loss


    def main_loop(self):

        for epoch in range(self.nb_epoch):
            start = time.time()

            while time.time()-start<self.minutes_per_epoch*60:
                X= self.data_generator()
                loss=self.train_step(X)
                self.train_losses.append(loss)

            print(f'duration of the {epoch}-th epoch: {time.time() - start:.2f} secs\n')
            plt.plot(self.train_losses)
            plt.show()


    def eval(self):
        X = data_generation_complex(self.batch_size, self.recurence_len, self.seq_len)
        X = X[:,1:]
        pred,_=self.predictor(X,False)
        ind=tf.range(self.seq_len-1)
        nb=5
        fig,axs=plt.subplots(nb,1)
        print(pred.shape)
        for i in range(nb):
            axs[i].plot(ind,X[i,:])
            axs[i].plot(ind, pred[i, :])

        plt.show()


def test_agent():
    seq_len=60
    batch_size=64

    data_generator=Data_generation_simple(batch_size,seq_len)
    predictor=PredictorViaAttention(seq_len-1,num_layers=2, d_model=8, num_heads=2, dff=12, rate=0.1)

    trainer=Trainer(predictor,data_generator)
    trainer.main_loop()
    trainer.eval()

if __name__ == "__main__":
    #test_multihead_attention()
    #test_data_generation_complex()
    test_agent()

