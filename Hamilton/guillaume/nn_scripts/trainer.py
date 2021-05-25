import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import time 
from scipy.integrate import solve_ivp

from hnn import HNN



plt.style.use("default")

class Trainer:
    def __init__(self, model, data, lr=1.e-3, wd=1.e-4):
        if not set( data.keys() ) == {'x', 'test_x', 'dx', 'test_dx'}:
            raise Exception("data not recognized")
        if model.__class__.__name__ not in ['HNN', 'sHNN']:
            raise TypeError('differentiable model should be a HNN or a sHNN instead of %s' % model.__class__.__name__)

            
        self.x = tf.constant(data['x'], dtype=tf.float32)
        self.test_x = tf.constant(data['test_x'], dtype=tf.float32)
        self.dx = tf.constant(data['dx'], dtype=tf.float32)
        self.test_dx = tf.constant(data['test_dx'], dtype=tf.float32)

        self.model = model
        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        self.test_loss = tf.keras.metrics.Mean(name = 'test_loss')

        self.optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)

        self.stats = {'train_loss': [], 'test_loss': [], 'epochs': 0}

        self.dir_name = 'model_' + time.ctime().replace(" ", "_") + '/'
        print('dir_name', self.dir_name)
        self.model_was_saved=False

    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.x)
            tape.watch(self.dx)
            hat_dx = self.model.time_derivative(self.x)
            loss = self.loss_object(self.dx, hat_dx)

        dtheta_loss = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(dtheta_loss, self.model.trainable_variables))
        self.train_loss(loss)

    @tf.function
    def test_step(self):
        hat_dx = self.model.time_derivative(self.x)
        t_loss = self.loss_object(self.dx, hat_dx)
        self.test_loss(t_loss)

    def train(self, epochs=1000):
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.test_loss.reset_states()

            self.train_step()
            self.test_step()

            loss = self.train_loss.result()
            self.stats['train_loss'].append(loss)
            t_loss = self.test_loss.result()
            self.stats['train_loss'].append(t_loss)


            if epoch%200 == 0:
                print(f'Epoch {epoch + 1}, '
                    f'Loss: {loss}, '
                    f'Test Loss: {t_loss}, ')
        
            self.stats['epochs'] += 1

#    def integrate_model(self, t_span, y0, t_eval=None):
#        def fun(t, np_x):
#            x = tf.reshape( tf.Variable(np_x, dtype=tf.float32), (1,2))
#            dx = tf.reshape( self.model.time_derivative(x), (2,))
#            return dx
#        return solve_ivp(fun=fun, t_span=t_span, t_eval=t_eval, y0=y0, rtol=1.e-12)

    def plot_loss(self, ax=None):
        if self.stats['epochs'] == 0:
            raise Exception("Model was not trained yet.")
        if ax is None:
            fig, ax = plt.subplots()

        ax.semilogy(self.stats['train_loss'], label='train_loss')
        ax.semilogy(self.stats['test_loss'], label='test_loss')
        ax.legend()

    def plot_pred(self, y0, trajectory, field, axs=None, figsize=(10,5)):
        arrow_width = 6e-3
        arrow_scale = 20
        if self.stats['epochs'] == 0:
            raise Exception("Model was not trained yet.")
        if axs is None:
            fig, axs = plt.subplots(1,2, figsize=figsize, sharey=True)

        _, coords, dcoords = trajectory

        N = coords.shape[0]
        point_colors = [(i/N, 0, 1-i/N) for i in range(N)]

        axs[0].scatter(coords[:,0], coords[:,1], label='data', s = 70, c=point_colors)
        axs[0].quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
                      cmap='gray_r', width=arrow_width, color=(.2,.2,.2), alpha=0.8) 
        axs[0].set_xlabel("$x$", fontsize=16)
        axs[0].set_ylabel("$\\frac{dx}{dt}$", rotation=0, fontsize=16)
        axs[0].set_title("Dynamics", fontsize=20)
        axs[0].legend(loc='upper right', fontsize=16)

        t_span = [0.0, 40.0]
        t_eval = np.linspace(t_span[0], t_span[-1], 600)
        sol_ivp = self.model.integrate_model(t_span=t_span, y0=y0, t_eval=t_eval)

        axs[1].quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1], 
                      cmap='gray_r', width=arrow_width, color=(.2,.2,.2), alpha=0.8)

        end0, end1 = [], []
        for i, l in enumerate(np.split(sol_ivp['y'].T, 10)):
            color = (float(i)/10, 0, 1-float(i)/10)
            axs[1].plot(np.insert(l[:,0], 0, end0), np.insert(l[:,1], 0, end1), color=color, linewidth=1.5)
            end0, end1 = l[-1,0], l[-1,1]

        axs[1].set_xlabel("$x$", fontsize=16)
        axs[1].set_title("Prediction", fontsize=20)
        plt.tight_layout()

if __name__ == "__main__":
    print('Hello, you\'re in %s' % __file__)