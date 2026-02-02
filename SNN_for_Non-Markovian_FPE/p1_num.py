# This code is a general framework applicable to all examples.
# Simply adjust the parameters for the chosen example to run different cases.
import os
import math
import time
import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from scipy import integrate

tf.disable_v2_behavior()
np.random.seed(1234)
tf.set_random_seed(1234)

class DLFPNN:
    def __init__(self, X, bound, step, layers, D):
        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.x_bound = bound[:, 0:1]
        self.t_bound = bound[:, 1:2]
        self.step = step
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        self.D = tf.constant(D, dtype=tf.float32)
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.b_x_tf = tf.placeholder(tf.float32, shape=[None, self.x_bound.shape[1]])
        self.b_t_tf = tf.placeholder(tf.float32, shape=[None, self.t_bound.shape[1]])
        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.u_bound = self.net_u(self.b_x_tf, self.b_t_tf)
        self.f_pred = self.net_f(self.x_tf, self.t_tf)
        self.normal = self.initial_error(self.x_tf, self.t_tf)
        self.loss_history = []

        self.loss1 = tf.reduce_mean(tf.square(self.f_pred))
        self.loss2 = tf.reduce_mean(tf.square(self.normal))
        self.loss3 = tf.reduce_mean(tf.square(self.u_bound))
        self.loss = 10 * self.loss1 + 5 * self.normal + 20*self.loss3
       
        self.lr = tf.Variable(0.01, trainable=False)  
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.nn.softplus(tf.add(tf.matmul(H, W), b))
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_t = tf.gradients(u, t)[0]
        f = u_t - self.D * u_xx + (u*(tf.cos(x)/6) + u_x*(tf.sin(x)/6+1))
        return f

    def initial_error(self, x, t):
        t_ini = [0.01]  
        total_loss = tf.constant(0.0, dtype=x.dtype)
        t_fixed = tf.ones_like(x) * tf.cast(t_ini, x.dtype) 
        u_pred = self.net_u(x, t_fixed)
        u_exact = self.exact_solution_func(x, t_fixed)
        err = u_pred - u_exact
        ini_loss = tf.reduce_mean(tf.square(err))
        return ini_loss
        

    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t,
                   self.b_x_tf: self.x_bound, self.b_t_tf: self.t_bound}
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.loss_history.append(loss_value)
            
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print(f'It: {it}, Loss: {loss_value:.3e}, Time: {elapsed:.2f}')
                start_time = time.time()

    def predict(self, X_star, bound):
        tf_dict = {self.x_tf: X_star[:, 0:1], self.t_tf: X_star[:, 1:2],
                   self.b_x_tf: bound[:, 0:1], self.b_t_tf: bound[:, 1:2]}
        u_star = self.sess.run(self.u_pred, {self.x_tf: X_star[:, 0:1], self.t_tf: X_star[:, 1:2]})
        loss_val = self.sess.run(self.loss, tf_dict)
        return u_star, loss_val

    def save_model(self, output_dir):
        model_dir = os.path.join(output_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(model_dir, "model.ckpt"))

    def exact_solution_func(self, x, t):
        D_value = self.D
        exact_u = 1.0 / tf.sqrt(4.0 * np.pi * D_value * t) * tf.exp(-tf.square(x-0.2-(tf.sin(0.2) / 6.0 + 1.0)* t) / (4.0 * D_value * t))
        return exact_u

    
if __name__ == "__main__":
    tf.reset_default_graph()
    layers = [2, 50, 50, 50, 1]
    D = 2
    step = 0.05
    x = np.arange(-2.8, 3.2, step)[:, np.newaxis]
    t_log = np.logspace(np.log10(0.001), np.log10(0.1), 30)
    t_linear = np.arange(0.1, 8.0, step*2)
    t = np.unique(np.concatenate((t_log, t_linear)))
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    bound = []
    for i in range(int(X_star.shape[0])):
        if X_star[i, 0] == x.min(0) or X_star[i, 0] == x.max(0):
            bound.append(X_star[i, :])
    bound = np.array(bound)
    output_dir = "./p1_num"
    os.makedirs(output_dir, exist_ok=True)

    model = DLFPNN(X_star, bound, step, layers, D)
    model.train(80000)
    model.save_model(output_dir)
