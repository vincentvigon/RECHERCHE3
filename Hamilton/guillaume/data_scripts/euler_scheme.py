###
# Note: This in an old script. Since, notation has changed p->q and u->p,
# the graphical outputs have been updated but the rest of the code remains
# unchanged (it works, so I do not modify it)
###


import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use("default")


def simple_euler_scheme_wave(final_time, dt, p0, u0, k, backward, figsize=(10,10)):
    t_list = []; p_list = []; u_list = []; H_list = []

    def Hamiltonian(p,u):
        return k * 0.5 * (p**2 + u**2)

    def update_lists(t, p, u):
        t_list.append(t)
        p_list.append(p)
        u_list.append(u)
        H_list.append( Hamiltonian(p,u) )

    tnow = 0.0
    denom = 1.0 + k**2 * dt**2
    pn, un = p0, u0
    update_lists(tnow,pn,un)

    while tnow < final_time:
        if not backward:
            pnp1 = pn + dt * k * un
            unp1 = un - dt * k * pn
        else:
            pnp1 = (pn + k * dt * un) / denom
            unp1 = (un - k * dt * pn) / denom
        tnow += dt
        pn, un = pnp1, unp1
        update_lists(tnow,pn,un)

    fig, axs = plt.subplots(2,2, figsize=figsize)

    axs[0,0].plot(t_list, p_list, "green")
    axs[0,0].set_title("Evolution of the position q(t)", fontsize = 14)
    axs[0,0].set_xlabel("t")
    axs[0,0].set_ylabel("q")
    axs[0,1].plot(t_list, u_list, "orange")
    axs[0,1].set_title("Evolution of the momentum p(t)", fontsize = 14)
    axs[0,1].set_xlabel("t")
    axs[0,1].set_ylabel("p")
    axs[1,0].plot(t_list, H_list, "red")
    axs[1,0].set_title("Evolution of the Hamiltonian H(q,p,t)", fontsize = 14)
    axs[1,0].set_xlabel("t")
    axs[1,0].set_ylabel("H")

    max_p = 1.2 * max(p_list)
    max_u = 1.2 * max(u_list)
    P = tf.linspace(-max_p, max_p, 11)
    U = tf.linspace(-max_u, max_u, 11)
    PP, UU = tf.meshgrid(P, U)

    with tf.GradientTape() as tape:
        tape.watch(PP)
        tape.watch(UU)
        H = 0.5 * k * (PP**2 + UU**2)
    dH_dP, dH_dU = tape.gradient(H, (PP, UU))

    q = axs[1,1].quiver(P, U, dH_dU, -dH_dP)

    axs[1,1].plot(p_list, u_list, "violet")
    axs[1,1].set_title("Phase portrait p(q)", fontsize = 14)
    axs[1,1].set_xlabel("q")
    axs[1,1].set_ylabel("p")
    fig.tight_layout()
    plt.show()


def tf_symplectic_euler_scheme_wave(final_time, dt, p0, u0, k, figsize=(10,10)):
    t_list = []; p_list = []; u_list = []; H_list = []

    def update_lists(t, p, u):
        t_list.append(t)
        p_list.append(p)
        u_list.append(u)
        H_list.append( 0.5 * k * (p**2 + u**2) )

    tnow = 0.0
    pn = tf.Variable(p0, dtype = tf.float64)
    un = tf.Variable(u0, dtype = tf.float64)
    update_lists(tnow,pn.numpy(),un.numpy())

    while tnow < final_time:
        with tf.GradientTape() as tape:
            H = 0.5 * k * (pn**2 + un**2)
        dH_du = tape.gradient(H, un)
        pn.assign_add( dt * dH_du )

        with tf.GradientTape() as tape:
            H = 0.5 * k * (pn**2 + un**2)
        dH_dp = tape.gradient(H, pn)
        un.assign_sub( dt * dH_dp )
        tnow += dt
        update_lists(tnow,pn.numpy(),un.numpy())

    fig, axs = plt.subplots(2,2, figsize=figsize)
    axs[0,0].plot(t_list, p_list, "green")
    axs[0,0].set_title("Evolution of the position q(t)", fontsize = 14)
    axs[0,0].set_xlabel("t")
    axs[0,0].set_ylabel("q")
    axs[0,1].plot(t_list, u_list, "orange")
    axs[0,1].set_title("Evolution of the momentum p(t)", fontsize = 14)
    axs[0,1].set_xlabel("t")
    axs[0,1].set_ylabel("p")
    axs[1,0].plot(t_list, H_list, "red")
    axs[1,0].set_title("Evolution of the Hamiltonian H(q,p,t)", fontsize = 14)
    axs[1,0].set_xlabel("t")
    axs[1,0].set_ylabel("H")


    max_p = 1.2 * max(p_list)
    max_u = 1.2 * max(u_list)
    P = tf.linspace(-max_p, max_p, 11)
    U = tf.linspace(-max_u, max_u, 11)
    PP, UU = tf.meshgrid(P, U)

    with tf.GradientTape() as tape:
        tape.watch(PP)
        tape.watch(UU)
        H = 0.5 * k * (PP**2 + UU**2)
    dH_dP, dH_dU = tape.gradient(H, (PP, UU))

    q = axs[1,1].quiver(P, U, dH_dU, -dH_dP)

    axs[1,1].plot(p_list, u_list, "violet")
    axs[1,1].set_title("Phase portrait p(q)", fontsize = 14)
    axs[1,1].set_xlabel("q")
    axs[1,1].set_ylabel("p")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    print('Hello, you\'re in %s' % __file__)