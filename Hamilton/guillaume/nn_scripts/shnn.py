import tensorflow as tf
from scipy.integrate import solve_ivp

from mlp import MLP

class sHNN(tf.keras.Model):
    def __init__(self, q_model=MLP(output_dim=1), p_model=MLP(output_dim=1)):
        
        if q_model.__class__.__name__!= 'MLP':
            raise TypeError('differentiable model should be a MLP instead of %s' % q_model.__class__.__name__)
        if p_model.__class__.__name__!= 'MLP':
            raise TypeError('differentiable model should be a MLP instead of %s' % p_model.__class__.__name__)
            
        super(sHNN, self).__init__()
        self.q_model = q_model
        self.p_model = p_model
        
    def call(self, x):
        q, p = tf.split(x, 2)
        return self.q_model(q) + self.p_model(p)
        
    @tf.function
    def time_derivative(self, x, t=None):
        q, p = tf.split(x, 2, axis=1)
        
        with tf.GradientTape() as qtape:
            qtape.watch(q)
            Hq = self.q_model(q)
        dq_Hq = qtape.gradient(Hq, q)
        
        with tf.GradientTape() as ptape:
            ptape.watch(p)
            Hp = self.p_model(p)
        dp_Hp = ptape.gradient(Hp, p)
        
        hat_dx = tf.stack([dp_Hp, tf.math.scalar_mul(-1.0, dq_Hq)], axis=1)
        return tf.squeeze(hat_dx) 
        
    def integrate_model(self, t_span, y0, t_eval=None):
        def fun(t, np_x):
            x = tf.reshape( tf.Variable(np_x, dtype=tf.float32), (1,2))
            dx = tf.reshape(self.time_derivative(x), (2,))
            return dx
        return solve_ivp(fun=fun, t_span=t_span, t_eval=t_eval, y0=y0, rtol=1.e-12) 
        
        
        