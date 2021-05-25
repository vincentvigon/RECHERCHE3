
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

"""
Malheureusement, la fonction reduce_power_generic 
 semble trop difficile à tracer. Cela bloque dans la méthode _der de l'agent
"""
@tf.function
def reduce_power_generic(a, out_dim):
    res = tf.TensorArray(dtype=tf.float32, size=out_dim)
    aa=a
    for i in tf.range(out_dim):
        res = res.write(i, tf.reduce_sum(aa,axis=1))
        aa*=a
    return tf.transpose(res.stack())

@tf.function
def reduce_power2(a):
    ra=tf.reduce_sum(a,axis=1)
    a2=a**2
    ra2=tf.reduce_sum(a2,axis=1)
    return tf.stack([ra,ra2],axis=1)

@tf.function
def reduce_power2_cross2(a):
    ra=tf.reduce_sum(a,axis=1)

    a2=a**2
    ra2=tf.reduce_sum(a2,axis=1)

    a_cross=a[:,:,tf.newaxis]*a[:,tf.newaxis,:]
    ra_cross = tf.reduce_sum(a_cross, axis=[1,2])

    return tf.stack([ra,ra2,ra_cross],axis=1)


def test_reduce_power():
    a=tf.constant([[1.,2,3,0],[1,1,1,1],[0,0,0,0]])
    b=reduce_power2(a)
    print("reduce_power2",b)
    c=reduce_power2_cross2(a)
    print("reduce_power2_cross2",c)






def test_argsort_np():
    vector = np.random.randint(0,10,10)
    aa = np.argsort(vector)
    aaa = np.argsort(aa)

    print(vector)
    b=vector[aa]
    print(b)
    c=b[aaa]
    print(c)


    vector = tf.constant([[10.,20,10,50],[3.,5,4,1]])
    aa = tf.argsort(vector,axis=1)
    aaa = tf.argsort(aa)

    print(vector)
    b = tf.gather(vector,aa)
    print(b)
    c = tf.gather(b,aaa)
    print(c)


@tf.function
def apply_ii(vectors, aa):
    res = tf.TensorArray(dtype=tf.float32,size=len(vectors))
    for i in tf.range(len(vectors)):
        res=res.write(i,tf.gather(vectors[i], aa[i]))
    #res = tf.stack(res)
    return res.stack()


#
# def apply_ii(vectors, aa):
#     res = []
#     for i in tf.range(len(vectors)):
#         res.append(tf.gather(vectors[i], aa[i]))
#     return tf.stack(res)



def test_argsort_tf():
    vectors = tf.constant([[1.,2,3,1], [ 1, 5, 4, 1]])
    aa=tf.argsort(vectors,axis=1)
    aaa=tf.argsort(aa,axis=1)
    res=apply_ii(vectors,aa)

    res2=apply_ii(res,aaa)

    print(res)
    print(res2)


def compacter(vectors, F):
    aa = tf.argsort(vectors)
    aaa = tf.argsort(aa)
    vectors_sorted=apply_ii(vectors,aa)
    F_vector_sorted=F(vectors_sorted)

    return apply_ii(F_vector_sorted,aaa)


def test_compacter():
    F=lambda v:10.*v
    res=compacter(tf.constant([[1.,2,3,1], [ 1, 5, 4, 1]]),F)
    print(res)

if __name__=="__main__":
    #test_argsort_tf()
    #test_compacter()
    test_reduce_power()
