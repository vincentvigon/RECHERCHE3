import tensorflow as tf
import utils

class MLP(tf.keras.Model):
    def __init__(self, input_dim=2, hidden_dim=200, output_dim=2,
                 nonlinearity='tanh'):
        super(MLP, self).__init__()

        nonlinearity = utils.choose_nonlinearity(nonlinearity)
        initializer = tf.keras.initializers.Orthogonal()

        self.Dense1 = tf.keras.layers.Dense(hidden_dim, activation=nonlinearity,
                                            kernel_initializer=initializer)
        self.Dense2 = tf.keras.layers.Dense(hidden_dim, activation=nonlinearity,
                                            kernel_initializer=initializer)
        self.Output = tf.keras.layers.Dense(output_dim, use_bias=False,
                                            kernel_initializer=initializer) #, weight_regularizer="l1")

    def __call__(self, x):
        x = self.Dense1(x)
        x = self.Dense2(x)
        return self.Output(x)
        
if __name__ == "__main__":
    print('Hello, you\'re in %s' % __file__)