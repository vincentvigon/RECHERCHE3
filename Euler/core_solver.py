# from tensorflow.python.types.core import Value
from Euler.param import Param, Projecter
from Euler.initial_conditions import *
import numpy as np
np.set_printoptions(precision=3, linewidth=100000)
pp = print

# pour comparer les performances
def compare(func_of_k):
    grid_nx = [100, 1000, 10000, 100000]
    grid_batch_size = [1, 32, 64, 128]

    def one_K(k):
        mat = np.zeros_float([len(grid_nx), len(grid_batch_size)])
        for i, nx in enumerate(grid_nx):
            for j, batch_size in enumerate(grid_batch_size):
                param = Param(nx, batch_size)
                t0 = time.time()
                func_of_k(k, param)
                mat[i, j] = time.time() - t0
        return mat
    mat_np = one_K(K("np", 32))
    mat_tf = one_K(K("tf", 64))
    df = pd.DataFrame(data=mat_np / mat_tf, columns=grid_batch_size, index=grid_nx)
    print(df)

import tensorflow as tf

def periodic_padding_flexible(tensor, axis,padding=1):
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """

    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax,p in zip(axis,padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right,middle,left], axis=ax)

    return tensor

def periodic_padding(W, k, pad):
    left = W[:, :pad, :]
    right = W[:, -pad:, :]
    return k.concatenate([right, W, left], axis=1)


def activation_for_relexive():
    x=tf.linspace(-3,3,100)
    y=tf.nn.elu(x-1)+1
    z=tf.nn.relu(x)
    plt.plot(x,y)
    plt.plot(x,z)
    plt.show()

def neumann_padding(W, k, pad):
    right_value = W[:, -1, :]
    left_value = W[:, 0, :]
    s = W.shape
    left_value_repeat = k.ones_float([s[0], pad, s[2]]) * left_value[:, k.newaxis, :]
    right_value_repeat = k.ones_float([s[0], pad, s[2]]) * right_value[:, k.newaxis, :]

    return k.concatenate([left_value_repeat, W, right_value_repeat], axis=1)

def make_positive_channels_0_2(A):
    A0,A1,A2=A[:,:,0],A[:,:,1],A[:,:,2]
    A0=tf.nn.elu(A0-1)+1
    A2=tf.nn.elu(A2-1)+1

    res= tf.stack([A0,A1,A2],axis=2)
    return res




def reflexive_padding(W, k, pad,for_solver:bool):
    right_value = W[:, -1, :]
    left_value = W[:, 0, :]
    s = W.shape
    left_value_repeat = k.ones_float([s[0], pad, s[2]]) * left_value[:, k.newaxis, :]
    right_value_repeat = k.ones_float([s[0], pad, s[2]]) * right_value[:, k.newaxis, :]

    left = W[:, 1:pad + 1, :] - left_value_repeat
    right = W[:, -1 - pad:-1, :] - right_value_repeat
    left = -left[:, ::-1, :]
    right = -right[:, ::-1, :]
    left += left_value_repeat
    right += right_value_repeat

    if for_solver:
        left=make_positive_channels_0_2(left)
        right=make_positive_channels_0_2(right)


    return k.concatenate([left, W, right], axis=1)

class Var:
    def __init__(self, W: np.ndarray, BC_solver:str,gamma:float  , k: K):
        """  toutes les variables _XXX sont de dimensions nx+2 """
        self.W = W
        self.k = k


        # la version prolongée de W
        if BC_solver == Param.BC_periodic:
            self.W_ = periodic_padding(W, k, 1)
        elif BC_solver == Param.BC_neumann:
            self.W_ = neumann_padding(W, k, 1)
        elif BC_solver == Param.BC_reflexive:
            self.W_ = reflexive_padding(W, k, 1,True)
        else:
            raise Exception("cette Boundary Condition pour le solver est inconnue:" + BC_solver)

        # des alias
        self.rho_ = self.W_[:, :, 0]  # density
        self.rhoV_ = self.W_[:, :, 1]  # moment
        self.E_ = self.W_[:, :, 2]  # energy

        self.V_ = self.rhoV_ / self.rho_

        self.rhoVV_ = self.rhoV_ * self.V_

        self.p_ = (gamma - 1) * (self.E_ - 1 / 2 * self.rhoVV_)  # pressure

        p_over_rho = self.p_ / self.rho_
        # pour éviter des Nan produite par de valeur négatives (résiduelle ?)
        p_over_rho = k.where_float(p_over_rho < 0., 0., p_over_rho)
        self.sound_ = k.sqrt(gamma * p_over_rho)  # sound_speed

        self.Flux_ = k.stack([self.rhoV_, self.rhoVV_ + self.p_, self.V_ * (self.E_ + self.p_)], axis=2)

    # nx+2

    def get_Flux(self):
        return self.Flux_

    def get(self, name, L_or_R):
        tensor = self.__dict__[name + "_"]
        if L_or_R == "L":
            return tensor[:, :-1]  # dim 2 ou 3
        elif L_or_R == "R":
            return tensor[:, 1:]  # dim 2 ou 3
        else:
            raise Exception("must be L or R")


    def get_augmentation(self):
        aug = self.k.stack([self.rho_, self.rhoV_, self.E_, self.V_, self.p_, self.sound_, self.Flux_[:, :, 1], self.Flux_[:, :, 2]], axis=2)
        return aug[:, 1:-1, :]


def Flux_mean(var):
    return (var.compute("Flux", "L") + var.compute("Flux", "R")) * 0.5

def HLLC_Wave_Speeds(Density_L, Density_R, Velocity_L, Velocity_R, Pressure_L, Pressure_R, Sound_L, Sound_R, gamma:float, k: K):
    Pressure_Mean = (Pressure_L + Pressure_R) / 2
    Density_Mean = (Density_L + Density_R) / 2
    Sound_Mean = (Sound_L + Sound_R) / 2

    Pressure_Star = k.maximum(k.zeros_like(Pressure_Mean),
                              Pressure_Mean - Density_Mean * Sound_Mean * (Velocity_R - Velocity_L) / 2)

    condition_L = Pressure_Star > Pressure_L
    q_L = k.where_float(condition_L, k.sqrt(1 + (gamma + 1) / (2 * gamma) * (Pressure_Star / Pressure_L - 1)), k.ones_like(Pressure_Star))

    condition_R = Pressure_Star > Pressure_R
    q_R = k.where_float(condition_R, k.sqrt(1 + (gamma + 1) / (2 * gamma) * (Pressure_Star / Pressure_R - 1)), k.ones_like(Pressure_Star))

    return (Velocity_L - Sound_L * q_L,
            Velocity_R + Sound_R * q_R)


def Flux_HLLC(var: Var, gamma:float, k: K):
    # vrai flux
    W_L = var.get("W", "L")
    W_R = var.get("W", "R")

    F_L = var.get("Flux", "L")
    F_R = var.get("Flux", "R")

    Density_L = W_L[:, :, 0]
    Density_R = W_R[:, :, 0]

    Momentum_L = W_L[:, :, 1]
    Momentum_R = W_R[:, :, 1]

    Velocity_L = var.get("V", "L")
    Velocity_R = var.get("V", "R")

    Pressure_L = var.get("p", "L")
    Pressure_R = var.get("p", "R")

    Sound_L = var.get("sound", "L")
    Sound_R = var.get("sound", "R")

    lamb_L, lamb_R = HLLC_Wave_Speeds(Density_L, Density_R, Velocity_L, Velocity_R, Pressure_L, Pressure_R, Sound_L, Sound_R, gamma, k)

    Lambda_Star = ((Pressure_R - Pressure_L
                  + Momentum_L * (lamb_L - Velocity_L)
                  - Momentum_R * (lamb_R - Velocity_R))
                  / (Density_L * (lamb_L - Velocity_L)
                   - Density_R * (lamb_R - Velocity_R)))

    Pressure_Star = Pressure_L + Density_L * (lamb_L - Velocity_L) * (Lambda_Star - Velocity_L)

    Density_L_Star = Density_L * (lamb_L - Velocity_L) / (lamb_L - Lambda_Star)
    W_L_Star_0 = Density_L_Star
    W_L_Star_1 = Density_L_Star * Lambda_Star
    W_L_Star_2 = Pressure_Star / (gamma - 1) + Density_L_Star * Lambda_Star**2 / 2
    W_L_Star = k.stack((W_L_Star_0, W_L_Star_1, W_L_Star_2), axis=2)

    Density_R_Star = Density_R * (lamb_R - Velocity_R) / (lamb_R - Lambda_Star)
    W_R_Star_0 = Density_R_Star
    W_R_Star_1 = Density_R_Star * Lambda_Star
    W_R_Star_2 = Pressure_Star / (gamma - 1) + Density_R_Star * Lambda_Star**2 / 2
    W_R_Star = k.stack((W_R_Star_0, W_R_Star_1, W_R_Star_2), axis=2)

    lamb_L = lamb_L[:, :, np.newaxis]
    Lambda_Star = Lambda_Star[:, :, np.newaxis]
    lamb_R = lamb_R[:, :, np.newaxis]

    F_L_Star = F_L + lamb_L * (W_L_Star - W_L)
    F_R_Star = F_R + lamb_R * (W_R_Star - W_R)

    zone_1 = (lamb_L >= 0)
    #todo: la zone_2 n'est pas utilisée
    zone_2 = k.logical_and((Lambda_Star >= 0), (lamb_L < 0))
    zone_3 = k.logical_and((lamb_R >= 0), (Lambda_Star < 0))
    zone_4 = (lamb_R < 0)

    numerical_Flux = k.where_float(zone_1, F_L, F_L_Star)
    numerical_Flux = k.where_float(zone_3, F_R_Star, numerical_Flux)
    numerical_Flux = k.where_float(zone_4, F_R, numerical_Flux)

    return numerical_Flux

def minmod(a, b, k: K):
    """ minmod limiter """
    c1 = k.logical_and(a > 0, b > 0)
    c2 = k.logical_and(a < 0, b < 0)

    limiter = k.where_float(c1, k.minimum(a, b), k.zeros_like(a))
    limiter = k.where_float(c2, k.maximum(a, b), limiter)

    return limiter


def SST_viscous(a, b, k: K):
    """ order 3 limiter without smoothness detection [ Schmidtmann Seibold Torrilhon 2015 ] """

    # zeros_N = k.zeros_float(a.shape)
    # limiter = np.copy(zeros_N)

    positive_a = a > 0
    positive_b = b > 0
    zeros = k.zeros_like(a)

    c1 = np.logical_and(positive_a, positive_b)
    c2 = np.logical_and(np.logical_not(positive_a), positive_b)
    c3 = np.logical_and(positive_a, np.logical_not(positive_b))
    c4 = np.logical_and(np.logical_not(positive_a), np.logical_not(positive_b))

    limiter = k.where_float(c1, k.maximum(zeros, k.minimum(k.minimum(2 * a, 3 / 2 * b), (a + 2 * b) / 3)), zeros)
    limiter = k.where_float(c2, k.maximum(zeros, k.minimum(- a, (a + 2 * b) / 3)), limiter)
    limiter = k.where_float(c3, k.minimum(zeros, k.maximum(- a, (a + 2 * b) / 3)), limiter)
    limiter = k.where_float(c4, k.minimum(zeros, k.maximum(k.maximum(2 * a, 3 / 2 * b), (a + 2 * b) / 3)), limiter)

    return limiter


@tf.function
def compute_solutions_order1(nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC_solver,gamma, W_init, k: K,is_coarse:bool):
    print("traçage de la fonction 'compute_solutions_order1' avec les arguments primitif:")
    print("\t\t nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC_solver,gamma,is_coarse=")
    print("\t\t",nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC_solver,gamma,is_coarse)
    print("\t\t et le tenseurs W_init de shape:",W_init.shape)

    """ res=k.zeros_float((nb_t,) + W_init.shape) ne fonctionne pas à cause de: EagerTensor' object does not support item assignment  """
    nx= nx_coarse if is_coarse else nx
    res = tf.TensorArray(tf.float32,size=nb_t,element_shape=[W_init.shape[0],nx,3],dynamic_size=False,clear_after_read=True)

    W = W_init  # inutile de faire une copie, elle est faites dans le Var() au moment du padding

    dt_over_dx= dt_over_dx_coarse if is_coarse else dt_over_dx

    for t in tf.range(nb_t):
        res=res.write(t,W)
        var = Var(W, BC_solver,gamma, k)
        Fnum = Flux_HLLC(var, gamma, k)
        dt_over_dx = dt_over_dx
        dFnum=Fnum[:, 1:, :] - Fnum[:, :-1, :]
        W = W - dt_over_dx * dFnum

    return res.stack()

def compute_solutions(param:Param, nb_t, W_init, is_coarse, k: K):

    dt_over_dx, dt_over_dx_coarse, nx, nx_coarse, BC_solver, gamma= param.dt_over_dx,param.dt_over_dx_coarse,param.nx,param.nx_coarse,param.BC_solver,param.gamma

    if is_coarse:
        assert W_init.shape[1] == param.nx_coarse
        return compute_solutions_order1(nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC_solver,gamma, W_init, k,True)
    else:
        assert W_init.shape[1] == param.nx
        if param.order == 1:
            return compute_solutions_order1(nb_t,dt_over_dx,dt_over_dx_coarse,nx,nx_coarse,BC_solver,gamma, W_init, k,False)
        else:
            raise ValueError("ordre>1: TODO")
        # elif param.order == 2:
        #     return compute_solutions_order2(param, nb_t, W_init, k)
        # elif param.order == 3:
        #     return compute_solutions_order3(param, nb_t, W_init, k)
        # else:
        #     raise ValueError("order > 3 is not supported in compute_solutions")



# pour les tests locaux uniquement
# def generate_fine_solutions(genParam: GenParam, param: Param, nb_t,k:K):
#     """ presque rien ici. """
#     generator = FuncGenerator(genParam, param, k)
#     W_init = generator.init_W()
#     return compute_solutions(param, nb_t, W_init, False, k)





def compare_with_projection_one_kind(is_periodic):

    k = K("tf", 32)
    if is_periodic:
        BC=Param.BC_periodic
    else:
        BC=Param.BC_neumann

    param = Param(600,  BC_solver=BC)

    if is_periodic:
        w_init = init_periodic(param,50)
    else:
        w_init = init_sod(param)#init_non_periodic(param,50)
    nb_t = 800
    w_init_coarse = Projecter(param.nx_ratio, 32).projection(w_init)

    res = compute_solutions(param, nb_t, w_init,  False,k)
    res_coarse = compute_solutions(param, nb_t, w_init_coarse,True,k)

    print(res.shape)

    fig, ax = plt.subplots(3, 2)

    title = "periodic" if is_periodic else "non periodic"
    fig.suptitle(title)

    for t in range(0, nb_t, nb_t // 10):
        color = "r" if t == 0 else "k"
        alpha = 1 if t == 0 else t / nb_t
        ax[0, 0].set_title("rho")
        ax[0, 0].plot(res[t, 0, :, 0], color, alpha=alpha)
        ax[0, 1].plot(res_coarse[t, 0, :, 0], color, alpha=alpha)
        ax[1, 0].set_title("rhoV")
        ax[1, 0].plot(res[t, 0, :, 1], color, alpha=alpha)
        ax[1, 1].plot(res_coarse[t, 0, :, 1], color, alpha=alpha)
        ax[2, 0].set_title("E")
        ax[2, 0].plot(res[t, 0, :, 2], color, alpha=alpha)
        ax[2, 1].plot(res_coarse[t, 0, :, 2], color, alpha=alpha)

    fig.tight_layout()
    plt.show()

def test_the_3_paddings():
    k = K('tf', 32)
    #W = k.arange_float(0, 9)[k.newaxis, :, k.newaxis]
    x=np.linspace(0,1,110,endpoint=False).astype(np.float32)
    y=tf.sin(x*2*np.pi)+(2*x+1)
    W=tf.stack([y,y,y],axis=1)
    W=W[tf.newaxis,:,:]
    print("W.shape",W.shape)

    pad=100
    W_neumann = neumann_padding(W, k, pad)
    W_periodic = periodic_padding(W, k, pad)
    W_reflexive = reflexive_padding(W, k, pad,True)
    assert W_neumann.shape == W_reflexive.shape == W_periodic.shape
    plt.plot(W_neumann[0, :, 0], label="neumann")
    plt.plot(W_periodic[0, :, 0], label="periodic")
    plt.plot(W_reflexive[0, :, 0], label="reflexive")
    plt.plot(np.zeros_like(x),color="k")
    plt.legend()
    plt.show()





def statistics_of_augmentation(kind):
    k = K("tf", 32)

    param = Param(100)

    batch_size = 20

    if kind=="periodic":
        BC = Param.BC_periodic
        w_init = init_periodic(param, batch_size)
    elif kind=="non-periodic":
        BC = Param.BC_reflexive
        w_init = init_non_periodic(param, batch_size)
    else :
        BC =Param.BC_neumann
        w_init = init_random_sod(param,batch_size)

    param.BC_model=BC
    nb_t = 800
    res = compute_solutions(param, nb_t, w_init, False, k)
    res = tf.reshape(res, [nb_t * param.nx * batch_size, 3])
    return res


def statistics_of_augmentation_all():
    res_pre=statistics_of_augmentation("periodic")
    res_non=statistics_of_augmentation("non-periodic")
    res_sod=statistics_of_augmentation("sod")

    res=tf.concat([res_pre,res_non,res_sod],axis=1)

    import seaborn as sns
    print(res.shape)

    sns.violinplot(data=res)
    plt.show()



if __name__ == '__main__':

    #activation_for_relexive()
    test_the_3_paddings()
    #compare_with_projection_one_kind(False)

    #statistics_of_augmentation_all()
