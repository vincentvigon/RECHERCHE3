import tensorflow as tf
import numpy as np

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
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

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output


class FF(tf.keras.layers.Layer):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()
        self.linear_1 = tf.keras.layers.Dense(d, activation='relu')
        self.linear_2 = tf.keras.layers.Dense(d, activation='relu')
        self.linear_3 = tf.keras.layers.Dense(d, activation='relu')
    def __call__(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.linear_3(self.linear_2(self.linear_1(x)))

class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.multihead = MultiHeadAttention(d, h)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.ff = FF(d)

    def __call__(self, x, y):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """

        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.ff(h))




class SetAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.mab = MultiHeadAttentionBlock(d, h)

    def __call__(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)


class InducedSetAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d: int, m: int, h: int):
        """
        Arguments:
            d: an integer, input dimension.
            m: an integer, number of inducing points.
            h: an integer, number of heads.
            ff1, ff2: modules, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()
        self.mab1 = MultiHeadAttentionBlock(d, h)
        self.mab2 = MultiHeadAttentionBlock(d, h)
        self.inducing_points = tf.random.normal(shape=(1, m, d))

    def __call__(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        b = tf.shape(x)[0]
        p = self.inducing_points
        p = tf.repeat(p, b, axis=0)  # shape [b, m, d]

        h = self.mab1(p, x)  # shape [b, m, d]
        return self.mab2(x, h)


class PoolingMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d: int, k: int, h: int):
        """
        Arguments:
            d: an integer, input dimension.
            k: an integer, number of seed vectors.
            h: an integer, number of heads.
            ff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()
        self.mab = MultiHeadAttentionBlock(d, h)
        self.seed_vectors = tf.random.normal(shape=(1, k, d))
        self.ff_s = FF(d)

    @tf.function
    def __call__(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, k, d]
        """
        b = tf.shape(z)[0]
        s = self.seed_vectors
        s = tf.repeat(s, b, axis=0)  # shape [b, k, d]
        return self.mab(s, self.ff_s(z))


class STEncoder(tf.keras.layers.Layer):
    def __init__(self, d=12, m=6, h=6):
        super().__init__()

        # Embedding part
        self.linear_1 = tf.keras.layers.Dense(d, activation='relu')

        # Encoding part
        self.isab_1 = InducedSetAttentionBlock(d, m, h)
        self.isab_2 = InducedSetAttentionBlock(d, m, h)

    def __call__(self, x):
        return self.isab_2(self.isab_1(self.linear_1(x)))


class STDecoder(tf.keras.layers.Layer):
    def __init__(self, out_dim, d=12, h=2, k=8):
        super().__init__()

        self.PMA = PoolingMultiHeadAttention(d, k, h)
        self.SAB = SetAttentionBlock(d, h)
        self.output_mapper = tf.keras.layers.Dense(out_dim)
        self.k, self.d = k, d

    def __call__(self, x):
        decoded_vec = self.SAB(self.PMA(x))
        decoded_vec = tf.reshape(decoded_vec, [-1, self.k * self.d])
        return tf.reshape(self.output_mapper(decoded_vec), (tf.shape(decoded_vec)[0],))


class BasicSetTransformer(tf.keras.Model):
    def __init__(self, encoder_d=4, m=3, encoder_h=2, out_dim=1, decoder_d=4, decoder_h=2, k=2):
        super().__init__()
        self.basic_encoder = STEncoder(d=encoder_d, m=m, h=encoder_h)
        self.basic_decoder = STDecoder(out_dim=out_dim, d=decoder_d, h=decoder_h, k=k)

    def call(self, x):
        enc_output = self.basic_encoder(x)  # (batch_size, set_len, d_model)
        return self.basic_decoder(enc_output)



def gen_max_dataset(dataset_size=100000, set_size=9, seed=0):

    """
    The number of objects per set is constant in this toy example
    """
    np.random.seed(seed)
    x = np.random.uniform(1, 100, (dataset_size, set_size))
    y = np.max(x, axis=1)
    x, y = np.expand_dims(x, axis=2), np.expand_dims(y, axis=1)
    return tf.cast(x, 'float32'), tf.cast(y, 'float32')


train_X, train_y = gen_max_dataset(dataset_size=100000, set_size=9, seed=1)
test_X, test_y = gen_max_dataset(dataset_size=15000, set_size=9, seed=3)
set_transformer = BasicSetTransformer()
set_transformer.compile(loss='mae', optimizer='adam')
set_transformer.fit(train_X, train_y, epochs=3)
predictions = set_transformer.predict(test_X)
print("MAE on test set is: ", np.abs(test_y - predictions).mean())