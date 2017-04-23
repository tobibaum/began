import time
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import OrderedDict

class Network(object):
    def __init__(self, data, **kwargs):
        self.layers = OrderedDict()
        if type(data) == dict:
            self.layers.update(data)
        else:
            self.layers['input'] = data
        self.data = data
        self.vars = []

        scope = kwargs.get('scope', '')
        reuse = kwargs.get('reuse', False)
        with tf.variable_scope(scope, reuse=reuse) as scope:
            self.setup()

    def setup(self):
        raise NotImplementedError

    def get_new_name(self, op):
        i = 0
        lay_base = op.__name__ + '_%d'
        while lay_base%i in self.layers:
            i += 1
        return lay_base%i

    def _layer(op):
        '''
        wrapper around layers to keep track
        '''
        def layer_wrapper(self, *args, **kwargs):
            name = kwargs.get('name', self.get_new_name(op))
            layer_input = self.layers.values()[-1]
            r = op(self, layer_input, name=name, *args)
            if type(r) == tuple:
                self.vars += r[1]
                r = r[0]
            self.layers[name] = r
            return self
        return layer_wrapper

    @property
    def output(self):
        return self.layers.values()[-1]

    @_layer
    def conv(self, input, k_w, k_h, c_o, s_w, name):
        '''
        input, kernel widht, kernel height, out channels, stride width, name
        '''
        c_i = input.get_shape()[-1]
        with tf.variable_scope(name) as scope:
            weight = tf.get_variable('weights', shape=[k_h, k_w, c_i, c_o],
                            initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', shape=[c_o],
                            initializer=tf.zeros_initializer())
            output = tf.nn.conv2d(input, weight, strides=[1, s_w, s_w, 1],
                    padding='SAME')
            output = tf.nn.bias_add(output, bias)
        return output, [weight, bias]

    @_layer
    def fc(self, input, num_out, name):
        c_i = input.get_shape()[-1]
        dim = 1
        for x in input.get_shape().as_list()[1:]:
            dim *= x
        with tf.variable_scope(name) as scope:
            weight = tf.get_variable('weights', shape=[dim, num_out],
                            initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('biases', shape=[num_out],
                            initializer=tf.zeros_initializer())
        input_reshape = tf.reshape(input, [-1, dim])
        output = tf.matmul(input_reshape, weight) + bias
        return output, [weight, bias]

    @_layer
    def relu(self, input, name):
        output = tf.nn.relu(input)
        return output

    @_layer
    def elu(self, input, name):
        output = tf.nn.elu(input)
        return output

    @_layer
    def pool(self, input, k_w, s_w, name):
        '''
        input, kernel_width, stride_width, name
        '''
        output = tf.nn.avg_pool(input, [1, k_w, k_w, 1], [1, s_w, s_w, 1],
                                padding='SAME', name=name)
        return output

    @_layer
    def upsample(self, input, s, name):
        in_size = int(input.get_shape()[1])
        out_size = in_size * s
        output = tf.image.resize_nearest_neighbor(input, [out_size, out_size],
                                                  name=name)
        return output

    @_layer
    def tanh(self, input, name):
        return tf.nn.tanh(input, name=name)

class encoder(Network):
    def setup(self):
        def block(n_filt, i, pool=True):
            (self.conv(3, 3, n_filt, 1, name='conv%d_a'%i)
                 .elu(name='elu%d_a'%i)
                 .conv(3, 3, n_filt, 1, name='conv%d_b'%i)
                 .elu(name='elu%d_b'%i))

            if pool:
                (self.conv(1, 1, n_filt, 1, name='conv%d_c'%i)
                     .pool(2, 2, name='pool%d'%i))

        n = 64
        h = 128
        block(n, 0)
        block(2*n, 1)
        block(3*n, 2)
        block(4*n, 3, pool=False)

        self.fc(128, name='fc_enc')

class generator(Network):
    def setup(self):
        n = 64

        self.fc(8*8*n, name='fc_gen')
        in_raw = self.output
        reshaped = tf.reshape(in_raw, [-1, 8, 8, n])
        self.layers['reshape'] = reshaped

        def up_block(n, i, up=True):
            (self.conv(3, 3, n, 1, name='conv%d_a'%i)
                 .elu(name='elu%d_a'%i)
                 .conv(3, 3, n, 1, name='conv%d_b'%i)
                 .elu(name='elu%d_b'%i))

            if up:
                self.upsample(2, name='upsample%d'%i)

        up_block(n, 0)
        up_block(n, 1)
        up_block(n, 2)
        up_block(n, 3, up=False)

        self.conv(1, 1, 3, 1, name='conv_img')
        self.tanh()

# hyper
lam = 0.001
gam = 0.5
lr = 0.0001
beta1 = 0.5
z_shape = 128
n_epochs = 10
bs = 16
seed = 0

# data
x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
z = tf.random_uniform([tf.shape(x)[0], z_shape], minval=-1, maxval=1)

# structure
gen = generator(z, scope='generator')

enc = encoder(gen.output, scope='encoder')
dec = generator(enc.output, scope='decoder')

enc_real = encoder(x, scope='encoder', reuse=True)
dec_real = generator(enc_real.output, scope='decoder', reuse=True)

# loss
k_t = tf.Variable(0.0, trainable=False, name='k_t')
loss_real = tf.reduce_mean(tf.abs(dec_real.output - x))
loss_fake = tf.reduce_mean(tf.abs(dec.output - gen.output))

loss_dis = loss_real - k_t * loss_fake
loss_gen = loss_fake

kt_op = k_t + lam * (gam*loss_real - loss_fake)
k_t = tf.clip_by_value(k_t, 0, 1)
conv_m = loss_real + tf.abs(gam*loss_real - loss_fake)

# optimizer
opt = tf.train.AdamOptimizer(lr, beta1)
dis_grad = opt.compute_gradients(loss_dis, var_list=enc.vars + dec.vars)
gen_grad = opt.compute_gradients(loss_dis, var_list=gen.vars)

dis_op = opt.apply_gradients(dis_grad)
gen_op = opt.apply_gradients(gen_grad)

train_ops = tf.group(dis_op, gen_op, kt_op)

# session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# load real data
data_raw = np.load('data.npy')
data = data_raw / 128. - 1

print 'loaded data, start training'

rng = np.random.RandomState(seed)
# train
for epoch in range(n_epochs):
    n_imgs = data.shape[0]
    n_batches = n_imgs / bs
    order = range(n_imgs)
    rng.shuffle(order)
    times = []
    for j in range(0, n_batches):
        imgs = data[j*bs:j*bs+bs]
        feed_dict = {x: imgs}
        ops = [train_ops, loss_gen, loss_dis, k_t, conv_m]

        t_step = time.time()
        run_res = sess.run(ops, feed_dict)
        t_step = time.time() - t_step
        times.append(t_step)

        if j%10 == 0:
            l_gen, l_dis, kt_val, conv_val = run_res[1:5]
            s_batch = np.mean(times)
            ex_s = bs / float(s_batch)
            out_text = '[%d: %d | %.2fex/s] -- '%(epoch, j, ex_s)
            out_text += 'gen: %.4f, dis: %.4f'%(l_gen, l_dis)
            out_text += ', k_t: %.4f, m: %.4f'%(kt_val, conv_val)
            times = []
            print out_text

        if j%50 == 0:
            imgs = sess.run(gen.output, feed_dict)
            img_grid = np.hstack([np.vstack(imgs[k:k+4]) \
                            for k in range(0, 16, 4)])
            img_out = (img_grid * 128 + 128).astype(np.uint8)
            Image.fromarray(img_out).save('results/%d_%d.png'%(epoch, j))
