import tensorflow as tf

def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = tf.keras.activations.tanh
    elif name == 'relu':
        nl = tf.keras.activations.relu
    elif name == 'sigmoid':
        nl = tf.keras.activations.sigmoid
    elif name == 'softplus':
        nl = tf.keras.activations.softplus
    elif name == 'selu':
        nl = tf.keras.activations.selu
    elif name == 'elu':
        nl = tf.keras.activations.elu
    elif name == 'swish':
        nl = tf.keras.activations.swish
    else:
        raise ValueError("nonlinearity not recognized")
    return nl
  

  
if __name__ == "__main__":
    print('Hello, you\'re in %s' % __file__)