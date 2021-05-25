import tensorflow as tf
from scipy.integrate import solve_ivp

from mlp import MLP

class HNN(tf.keras.Model):
    def __init__(self, differentiable_model=MLP(), input_dim=2, baseline=False):
        if differentiable_model.__class__.__name__!= 'MLP':
            raise TypeError('differentiable model should be a MLP instead of %s' % differentiable_model.__class__.__name__)
        super(HNN, self).__init__()

        self.baseline = baseline
        self.differentiable_model = differentiable_model

    def __call__(self, x):
        return self.differentiable_model(x)
        
    @tf.function
    def time_derivative(self, x, t=None):
        if self.baseline:
            return self.differentiable_model(x)

        with tf.GradientTape() as tape:
            tape.watch(x)
            H = self.differentiable_model(x)
        dqp_H = tape.gradient(H, x)
        dH_dq, dH_dp = tf.split(dqp_H, 2, axis=1)
        hat_dx = tf.stack([dH_dp, tf.math.scalar_mul(-1.0, dH_dq)], axis=1)
        
        return tf.squeeze(hat_dx) # hat_dx[:,:,0]
        
    def integrate_model(self, t_span, y0, t_eval=None):
        def fun(t, np_x):
            x = tf.reshape( tf.Variable(np_x, dtype=tf.float32), (1,2))
            dx = tf.reshape(self.time_derivative(x), (2,))
            return dx
        return solve_ivp(fun=fun, t_span=t_span, t_eval=t_eval, y0=y0, rtol=1.e-12) 
        
if __name__ == "__main__":
    print('Hello, you\'re in %s' % __file__)