import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use("default")

class data_generator:
    def __init__(self, system, **kwargs):
        assert system in ['spring', 'pendulum'], 'Unknown physical system'
        
        m = kwargs.get('m', 1.0)
        g = kwargs.get('g', 3.0)
        l = kwargs.get('l', 1.0)
        k = kwargs.get('k', 1.0)
        
        #- in case we want to retrieve the analytical solution
        self.pulse = np.sqrt( k / m )
        
        print('Parameters: \n\t system: %s \n\t m = %.3f \n\t g = %.3f '\
              '\n\t l = %.3f \n\t k = %.3f' % (system, m, g, l, k) )
        
        if system == 'spring':
            def hamiltonian_func(coords):
                q, p = np.split(coords, 2)
                return k * q**2 / 2.0 + p**2 / 2.0 / m
        
            def dynamic_func(t, coords):
                q, p = np.split(coords, 2)
                dH_dq = k * q # = \dfrac{\partial H}{\partial q} = k * q
                dH_dp = p / m # = \dfrac{\partial H}{\partial p} = p / m
                return dH_dp, -dH_dq
        
        if system == 'pendulum':
            def hamiltonian_func(coords):
                q, p = np.split(coords, 2)
                # return p**2 / 2.0 / m / l**2 - m * g * l * np.cos(q)
                return 2.0 * m * g * l * (1.0 - np.cos(q.astype(np.float))) + l**2 * p**2 / 2.0 / m 
    
            def dynamic_func(t, coords):
                q, p = np.split(coords, 2)
                dH_dq =  2.0 * m * g * l * np.sin(q.astype(np.float)) # = \dfrac{\partial H}{\partial q} = 2*m*g*l*sin(q)
                dH_dp = l**2 * p / m # = \dfrac{\partial H}{\partial p} = l**2 * p / m
                return dH_dp, -dH_dq 
        
        self.hamiltonian_func = hamiltonian_func
        self.dynamic_func = dynamic_func
        self.system = system

    def get_trajectory(self, t_span=[0.0, 8.0], timescale=None, radius=None, y0=None, noise_std=0.1, plot=False, figsize=(10,10),
                       polar_harm=False, polar_harm_amp=0.1, polar_harm_freq=16.0, post_noise_target=False, exact_grad=False):
        if timescale is None and self.system == 'spring':
            timescale = 10;
        elif timescale is None and self.system == 'pendulum':
            timescale = 15;
        
        t_eval, dt = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])), retstep =True)
        no_timestep = t_eval.shape[0]
        
        if exact_grad:    
            def derive_coords(t, coordsp1, coords, coordsm1):
                return self.dynamic_func(t, coords)
        else:
            def derive_coords(t,coordsp1, coords, coordsm1):
                return (coordsp1 - coordsm1) / dt / 2.0
        
        # get initial state
        if y0 is None and self.system == 'spring':
            y0 = np.random.rand(2)*2 - 1.0
        elif y0 is None and self.system == 'pendulum':
            y0 = np.random.rand(2)*2 - 1.0

        if radius is None and self.system == 'spring':
            radius = np.random.rand() * 2.0 - 1.0
        elif radius is None and self.system == 'pendulum':
            radius = np.random.rand() + 1.3

        y0 = y0 / np.sqrt( ( y0**2 ).sum() ) * radius 

        # symplectic scheme
        coords = np.empty((t_eval.shape[0], 2))
        coords[0,:] = y0
        
        dcoords = np.empty_like(coords)
        qn, pn = y0[0], y0[1]

        for i in range(1,no_timestep):
            coords[i,:] = coords[i-1,:]

            _, dp_dt = self.dynamic_func(None, coords[i,:])
            coords[i,1] += dt * (-dp_dt)

            dq_dt, _ = self.dynamic_func(None, coords[i,:])
            coords[i,0] -= dt * dq_dt
                
        if polar_harm:        
            for i in range(no_timestep):
                r = np.sqrt(coords[i,0]**2 + coords[i,1]**2)
                theta = np.arctan2(coords[i,1], coords[i,0])
                coords[i,0] = (r + polar_harm_amp * np.sin(2.0 * np.pi * polar_harm_freq * theta)) * np.cos(theta)
                coords[i,1] = (r + polar_harm_amp * np.sin(2.0 * np.pi * polar_harm_freq * theta)) * np.sin(theta)

        if plot:
            H = np.empty(no_timestep)
            for i in range(no_timestep):
                H[i] = self.hamiltonian_func(coords[i,:])
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            axs[0,0].plot(t_eval, coords[:,0], "green")
            axs[0,0].set_title("q(t)", fontsize = 20)
            axs[0,1].plot(t_eval, coords[:,1], "orange")
            axs[0,1].set_title("p(t)", fontsize = 20)
            axs[1,0].plot(t_eval, H, "red")
            axs[1,0].set_title("H(t)", fontsize = 20)
            axs[1,1].plot(coords[:,0], coords[:,1], "blue")
            axs[1,1].set_title("p(q)", fontsize = 20)

        # compute target
        if not post_noise_target:
            dq_dt, dp_dt = derive_coords(None, 2.0 * coords[1,:], coords[0,:], 2.0 * coords[0,:])
            dcoords[0,:] = np.squeeze( np.array([dq_dt, dp_dt]) )
            for i in range(1, no_timestep - 1):
                dq_dt, dp_dt = derive_coords(None, coords[i + 1,:], coords[i,:], coords[i - 1,:])
                dcoords[i,:] = np.squeeze( np.array([dq_dt, dp_dt]) )
            dq_dt, dp_dt = derive_coords(None, 2.0 * coords[no_timestep - 1,:], coords[no_timestep - 1,:], 2.0 * coords[no_timestep - 2,:])
            dcoords[0,:] = np.squeeze( np.array([dq_dt, dp_dt]) )

        # add noise
        coords += np.random.randn(coords.shape[0], coords.shape[1]) * noise_std
        
        # compute target
        if post_noise_target:
            dq_dt, dp_dt = derive_coords(None, 2.0 * coords[1,:], coords[0,:], 2.0 * coords[0,:])
            dcoords[0,:] = np.squeeze( np.array([dq_dt, dp_dt]) )
            for i in range(1, no_timestep - 1):
                dq_dt, dp_dt = derive_coords(None, coords[i + 1,:], coords[i,:], coords[i - 1,:])
                dcoords[i,:] = np.squeeze( np.array([dq_dt, dp_dt]) )
            dq_dt, dp_dt = derive_coords(None, 2.0 * coords[no_timestep - 1,:], coords[no_timestep - 1,:], 2.0 * coords[no_timestep - 2,:])
            dcoords[0,:] = np.squeeze( np.array([dq_dt, dp_dt]) )

        if plot:
            for i in range(no_timestep):
                H[i] = self.hamiltonian_func(coords[i,:])
            axs[0,0].scatter(t_eval, coords[:,0], color="green", alpha=0.4)
            axs[0,1].scatter(t_eval, coords[:,1], color="orange", alpha=0.4)
            axs[1,0].scatter(t_eval, H, color="red", alpha=0.4)
            axs[1,1].scatter(coords[:,0], coords[:,1], color="blue", alpha=0.4)

            max_q = np.amax( coords[:,0] )
            max_p = np.amax( coords[:,1] )

            Q = np.linspace(-max_q, max_q, 10)
            P = np.linspace(-max_p, max_p, 10)
            QQ, PP = np.meshgrid(Q, P)
            dQ_dt, dP_dt = self.dynamic_func(None, np.array( [QQ.flatten(), PP.flatten()] ))

            axs[1,1].quiver(QQ, PP, dQ_dt, dP_dt)
    
        return t_eval, coords, dcoords

    def get_dataset(self, seed=0, samples=50, test_split=0.5, noise_std=0.1, timescale=None,
                       polar_harm=False, polar_harm_amp=0.1, polar_harm_freq=16.0, post_noise_target=False, t_span=[0.0, 8.0],
                       exact_grad=False):
        data = {'meta': locals()}

        # sample of random inputs
        np.random.seed(seed)
        coords_list, dcoords_list = [], []
        for s in range(samples):
            t_eval , coords, dcoords = self.get_trajectory(noise_std=noise_std, timescale=timescale, polar_harm=polar_harm,
                                                           polar_harm_amp=polar_harm_amp, polar_harm_freq=polar_harm_freq,
                                                           post_noise_target=post_noise_target, t_span=t_span, exact_grad=exact_grad,
                                                           timescale=timescale)
            coords_list.append( coords ) 
            dcoords_list.append( dcoords )
        data['x'] = np.concatenate(coords_list)
        data['dx'] = np.concatenate(dcoords_list)

        # make a train/test split
        split_ix = int(len(data['x']) * test_split)
        split_data = {}
        for k in ['x', 'dx']:
            split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
        data = split_data
        return data
                
    def get_field(self, xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=10):
        field = {'meta': locals()}
        Q = np.linspace(xmin, xmax, gridsize)
        P = np.linspace(ymin, ymax, gridsize)
        # QQ, PP = np.meshgrid(P, Q) 
        QQ, PP = np.meshgrid(Q,P)
        flattened_QP = np.array( [QQ.flatten(), PP.flatten()] )
        dQ_dt, dP_dt = self.dynamic_func(None, flattened_QP)
        field['x'] = flattened_QP.T
        field['dx'] = np.array( [dQ_dt.flatten(), dP_dt.flatten()] ).T
        return field

    def integrate_dynamic(self, t_span, y0, t_eval=None):
        def func(t, x):
            dq_dt, dp_dt = self.dynamic_func(t, x)
            return np.array([dq_dt[0], dp_dt[0]])
        return solve_ivp(fun=func, t_span=t_span, y0=y0, 
                         t_eval=t_eval, rtol=1.e-12)

if __name__ == "__main__":
    print('Hello, you\'re in %s' % __file__)