import numpy as np
import tensorflow as tf
import time
import pandas as pd

pp = print

# Kontext


class K:

    def __init__(self, kind: str, precision: int):
        self.kind = kind
        self.precision = precision

        assert kind == "np" or kind == "tf"
        assert precision == 32 or precision == 64
        i = 0 if kind == "np" else 1

        self.name = ("numpy", "tensorflow")[i]

        if precision == 32:
            self.float = (np.float32, tf.float32)[i]
            self.int = (np.int32, tf.int32)[i]
        else:
            self.float = (np.float64, tf.float64)[i]
            self.int = (np.int32, tf.int32)[i]

        self.pad = (np.pad, tf.pad)[i]
        self.stack = (np.stack, tf.stack)[i]
        self.minimum = (np.minimum, tf.minimum)[i]
        self.maximum = (np.maximum, tf.maximum)[i]
        self.max = (np.max, tf.reduce_max)[i]
        self.min = (np.min, tf.reduce_min)[i]
        self.sum = (np.sum, tf.reduce_sum)[i]
        self.mean = (np.mean, tf.reduce_mean)[i]
        self.abs = (np.abs, tf.abs)[i]
        self.cos = (np.cos, tf.cos)[i]
        self.sin = (np.sin, tf.sin)[i]
        self.exp = (np.exp, tf.exp)[i]

        self.newaxis = (np.newaxis, tf.newaxis)[i]
        self.reshape = (np.reshape, tf.reshape)[i]
        self.sqrt = (np.sqrt, tf.sqrt)[i]
        self.logical_and = (np.logical_and, tf.logical_and)[i]
        self.logical_not = (np.logical_not, tf.logical_not)[i]
        self.logical_or = (np.logical_or, tf.logical_or)[i]
        self.concatenate = (np.concatenate, tf.concat)[i]
        self.set_seed = (np.random.seed, tf.random.set_seed)[i]

        """Les méthodes suivantes générent des tableau. Attention au dtype"""
        if i == 0:  # numpy
            self.ones_float = lambda shape: np.ones(shape=shape, dtype=self.float)
            self.zeros_float = lambda shape: np.zeros(shape=shape, dtype=self.float)
            self.ones_like = lambda values: np.ones_like(values)
            self.zeros_like = lambda values: np.zeros_like(values)
            self.linspace_float = lambda xmin, xmax, nx: np.linspace(xmin, xmax, nx, dtype=self.float)
            self.arange_float = lambda deb, end, delta=1: np.arange(deb, end, delta, dtype=self.float)
            # le seul tenseur avec des entiers
            self.arange_int = lambda deb, end, delta=1: np.arange(deb, end, delta, dtype=self.int)

            self.where_float = lambda condition, x, y: np.where(condition, x, y).astype(dtype=self.float)
            self.array_float = lambda values: np.array(values, dtype=self.float)
            # numpy et tensorflow n'ont pas exactement les même algos de génération aléatoire
            # attention chez numpy shape->size, et pas de dtype
            self.random_uniform_float = lambda minval, maxval, shape: np.random.uniform(low=minval, high=maxval,
                                                                                        size=shape).astype(self.float)

        else:  # tensorflow
            self.ones_float = lambda shape: tf.ones(shape=shape, dtype=self.float)
            self.zeros_float = lambda shape: tf.zeros(shape=shape, dtype=self.float)
            self.ones_like = lambda values: tf.ones_like(values)
            self.zeros_like = lambda values: tf.zeros_like(values)
            self.linspace_float = lambda xmin, xmax, nx: tf.cast(tf.linspace(xmin, xmax, nx), dtype=self.float)
            self.arange_float = lambda deb, end, delta=1: tf.range(deb, end, delta=delta, dtype=self.float)
            self.arange_int = lambda deb, end, delta=1: tf.range(deb, end, delta=delta, dtype=self.int)
            self.where_float = lambda condition, x, y: tf.cast(tf.where(condition, x, y), dtype=self.float)
            self.array_float = lambda values: tf.constant(values, dtype=self.float)
            self.random_uniform_float = lambda minval, maxval, shape: tf.random.uniform(minval=minval, maxval=maxval,
                                                                                        shape=shape, dtype=self.float)
        # tf.pi n'est pas défini
        self.pi = np.pi

    def __str__(self):
        return self.kind + str(self.precision)

    def __repr__(self):
        print(self.__str__())

    def convert(self, tensor, to_float=True):
        if self.kind == "tf":
            if to_float:
                return tf.cast(tensor, self.float)
            else:
                return tf.constant(tensor, self.int)
        if self.kind == "np":
            if to_float:
                return np.array(tensor, self.float)
            else:
                return np.array(tensor, self.int)

    def check_mine(self, tensor, raiseException=True):
        is_tf = hasattr(tensor, "numpy")
        if is_tf and self.kind == "np":
            if raiseException:
                raise Exception("le tenseur en entrée est de type: tf alors que le contexte est en np")
            else:
                return False
        if not is_tf and self.kind == "tf":
            if raiseException:
                raise Exception("le tenseur en entrée est de type: np alors que le contexte est en tf")
            else:
                return False

        if not (tensor.dtype == self.float or tensor.dtype == self.int):
            if raiseException:
                raise Exception("le tenseur en entrée a comme dtype:" + str(
                    tensor.dtype) + " alors que la précision du contexte est:" + str(self))
            else:
                return False

        return True


class Ks:

    @staticmethod
    def format_time(t):
        # 0.000123=>123 mus
        sec = int(t)
        millis = int((t - sec) * 1000)
        micros = int(((t - sec) * 1000 - millis) * 1000)
        res = ""
        if sec > 0:
            res += str(sec) + "s"
        if sec < 10:
            if millis > 0:
                res += str(millis) + "ms"
            if millis < 10:

                if micros > 0:
                    res += str(micros) + "μs"
            return res

    @staticmethod
    def compare_tf_atDecorated(func_of_k):
        print("compare tf and @tf_function (decoration) for function:", func_of_k.__name__)

        tf.config.run_functions_eagerly(True)
        ti0 = time.time()
        func_of_k(K("tf", 32))
        time_tf_32 = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("tf", 64))
        time_tf_64 = time.time() - ti0

        tf.config.run_functions_eagerly(False)
        ti0 = time.time()
        func_of_k(K("tf", 32))
        time_tf_32_at = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("tf", 64))
        time_tf_64_at = time.time() - ti0

        mat = [
            [Ks.format_time(time_tf_64), np.round(time_tf_32 / time_tf_64, 2)],
            [np.round(time_tf_64_at / time_tf_64, 2), np.round(time_tf_32_at / time_tf_64, 2)],
        ]
        df = pd.DataFrame(data=mat, columns=[64, 32], index=["tf", "@tf"])
        return df

    @staticmethod
    def compare_np_tf_at(func_of_k):

        print("compare np, tf and @tf.function for function:", func_of_k.__name__)

        ti0 = time.time()
        func_of_k(K("np", 64))
        time_np_64 = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("np", 32))
        time_np_32 = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("tf", 32))
        time_tf_32 = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("tf", 64))
        time_tf_64 = time.time() - ti0

        tf_function_obj = tf.function(lambda: func_of_k(K("tf", 32)))
        concrete_function = tf_function_obj.get_concrete_function()
        ti0 = time.time()
        concrete_function()
        time_tf_32_at = time.time() - ti0

        tf_function_obj = tf.function(lambda: func_of_k(K("tf", 64)))
        concrete_function = tf_function_obj.get_concrete_function()
        ti0 = time.time()
        concrete_function()
        time_tf_64_at = time.time() - ti0

        mat = [[Ks.format_time(time_np_64), np.round(time_np_32 / time_np_64, 2)],
               [np.round(time_tf_64 / time_np_64, 2), np.round(time_tf_32 / time_np_64, 2)],
               [np.round(time_tf_64_at / time_np_64, 2), np.round(time_tf_32_at / time_np_64, 2)],
               ]
        df = pd.DataFrame(data=mat, columns=[64, 32], index=["np", "tf", "@tf"])
        return df

    @staticmethod
    def compare_tf_at(func_of_k):

        print("compare tf and @tf.function (no-decoration) for function:", func_of_k.__name__)

        ti0 = time.time()
        func_of_k(K("tf", 32))
        time_tf_32 = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("tf", 64))
        time_tf_64 = time.time() - ti0

        tf_function_obj = tf.function(lambda: func_of_k(K("tf", 32)))
        concrete_function = tf_function_obj.get_concrete_function()
        ti0 = time.time()
        concrete_function()
        time_tf_32_at = time.time() - ti0

        tf_function_obj = tf.function(lambda: func_of_k(K("tf", 64)))
        concrete_function = tf_function_obj.get_concrete_function()
        ti0 = time.time()
        concrete_function()
        time_tf_64_at = time.time() - ti0

        mat = [
            [Ks.format_time(time_tf_64), np.round(time_tf_32 / time_tf_64, 2)],
            [np.round(time_tf_64_at / time_tf_64, 2), np.round(time_tf_32_at / time_tf_64, 2)],
        ]
        df = pd.DataFrame(data=mat, columns=[64, 32], index=["tf", "@tf"])
        return df

    @staticmethod
    def compare_tf(func_of_k):

        print("compare tf  for function:", func_of_k.__name__)

        ti0 = time.time()
        func_of_k(K("tf", 32))
        time_tf_32 = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("tf", 64))
        time_tf_64 = time.time() - ti0

        mat = [
            [Ks.format_time(time_tf_64), np.round(time_tf_32 / time_tf_64, 2)]
        ]
        df = pd.DataFrame(data=mat, columns=[64, 32], index=["tf"])
        return df

    @staticmethod
    def compare_np_tf(func_of_k):
        print("compare np and tf for function:", func_of_k.__name__)

        ti0 = time.time()
        func_of_k(K("np", 64))
        time_np_64 = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("np", 32))
        time_np_32 = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("tf", 32))
        time_tf_32 = time.time() - ti0

        ti0 = time.time()
        func_of_k(K("tf", 64))
        time_tf_64 = time.time() - ti0

        mat = [[Ks.format_time(time_np_64), np.round(time_np_32 / time_np_64, 2)],
               [np.round(time_tf_64 / time_np_64, 2), np.round(time_tf_32 / time_np_64, 2)]
               ]
        df = pd.DataFrame(data=mat, columns=[64, 32], index=["np", "tf"])
        return df


def equalities(k: K):
    dico_res = {}

    C = k.array_float([-1, 1])
    Z = k.zeros_float([2, 3])
    O = k.ones_float([2, 3])
    A = k.where_float(C[:, tf.newaxis] > 0, Z, O)
    dico_res["where"] = A

    W = k.zeros_float([32, 10, 3])
    dico_res["pad"] = k.pad(W, [[0, 0], [1, 1], [0, 0]], "reflect")

    a = k.ones_float([32, 10, 3])
    b = k.ones_float([32, 10, 3])
    c = k.ones_float([32, 10, 3])
    abc = k.stack([a, b, c], axis=2)
    dico_res["stack"] = abc
    dico_res["minimum"] = k.minimum([1, 2, 3], [3, 2, 1])
    dico_res["random_uniform.shape"] = k.array_float(k.random_uniform_float(shape=(5, 5), minval=0, maxval=1).shape)

    return dico_res


def test_equalities(precision):
    dico_np = equalities(K("np", precision))
    dico_tf = equalities(K("tf", precision))
    for key, value in dico_np.items():
        diff = np.sum(np.abs(dico_np[key] - dico_tf[key]))
        if diff > 1e-6:
            raise Exception("problem with:", key)


def test_compare():
    def oper(k: K):
        size = 1000
        A = k.ones_float([size, size])
        for i in range(5):
            A = A * A

    print(Ks.compare_np_tf_at(oper))

    def external_loop(k: K):
        @tf.function
        def sub_oper(A):
            A = k.sin(A) * k.cos(A)
            return A

        size = 1000
        A = k.ones_float([size, size])
        for _ in range(200):
            A = sub_oper(A)

    def internal_loop(k: K):
        @tf.function
        def sub_oper(A):
            for _ in range(20):
                A = k.sin(A) * k.cos(A)
            return A

        size = 1000
        A = k.ones_float([size, size])
        for _ in range(40):
            A = sub_oper(A)

    print(Ks.compare_tf_atDecorated(external_loop))
    print(Ks.compare_tf_atDecorated(internal_loop))


def test_check():
    for kind in ["np", "tf"]:
        for precision in [32, 64]:
            for kind2 in ["np", "tf"]:
                for precision2 in [32, 64]:
                    k = K(kind, precision)
                    k2 = K(kind2, precision2)
                    tensor = k2.ones_float([2, 2])

                    ok = k.check_mine(tensor, False)
                    if ok and (kind2 != kind or precision != precision2):
                        print(kind, kind2, precision, precision2)
                        raise Exception("problem")

                    if not ok and (kind2 == kind and precision == precision2):
                        print(kind, kind2, precision, precision2)
                        raise Exception("problem: no ok while it might be ok")

                    tensor_conv = k.convert(tensor)
                    k.check_mine(tensor_conv)


def test_format_time():
    t = 1.234
    print(Ks.format_time(t))

    t = 0.234_345
    print(Ks.format_time(t))

    t = 0.003_123
    print(Ks.format_time(t))


if __name__ == "__main__":
    test_compare()

    # test_format_time()

    assert tf.float32 == np.float32
    test_check()
    test_equalities(32)
    test_equalities(64)
