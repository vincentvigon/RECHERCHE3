import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(MyLayer,self).__init__()

        self.linear1=tf.keras.layers.Dense(5)
        self.linear2=tf.keras.layers.Dense(5)

    def __call__(self,x):

        y1=self.linear1(x)
        y2=self.linear2(x)

        return y1*y2


class MyModel(tf.keras.layers.Layer):

    def __init__(self):
        super(MyModel,self).__init__()

        self.layer1=MyLayer()
        self.layer2=MyLayer()

    def __call__(self,x):
        y=self.layer1(x)
        z=self.layer2(y)
        return z


def test():
    myModel=MyModel()
    x=tf.random.uniform([10,3])

    print(myModel(x).shape)
    print(myModel.trainable_variables)

    print(myModel.get_weights())


test()



