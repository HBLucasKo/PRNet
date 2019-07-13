

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope
import numpy as np


def resBlock(x, num_outputs, kernel_size = 4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope=None):
    assert num_outputs%2==0 #num_outputs must be divided by channel_factor(2 here)
    with tf.variable_scope(scope, 'resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride, 
                        activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=kernel_size, stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)

        x += shortcut       
        x = normalizer_fn(x)
        x = activation_fn(x)
    return x


class resfcn256(object):
    def __init__(self, resolution_inp = 256, resolution_op = 256, channel = 3, name = 'resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training = True):
        with tf.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu, 
                                     normalizer_fn=tcl.batch_norm, 
                                     biases_initializer=None, 
                                     padding='SAME',
                                     weights_regularizer=tcl.l2_regularizer(0.0002)):
                    size = 16  
                    # x: s x s x 3
                    se1 = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1) # 256 x 256 x 16
                    se2 = resBlock(se1, num_outputs=size * 2, kernel_size=4, stride=2) # 128 x 128 x 32
                    se3 = resBlock(se2, num_outputs=size * 2, kernel_size=4, stride=1) # 128 x 128 x 32
                    se4 = resBlock(se3, num_outputs=size * 4, kernel_size=4, stride=2) # 64 x 64 x 64
                    se5 = resBlock(se4, num_outputs=size * 4, kernel_size=4, stride=1) # 64 x 64 x 64
                    se6 = resBlock(se5, num_outputs=size * 8, kernel_size=4, stride=2) # 32 x 32 x 128
                    se7 = resBlock(se6, num_outputs=size * 8, kernel_size=4, stride=1) # 32 x 32 x 128
                    se8 = resBlock(se7, num_outputs=size * 16, kernel_size=4, stride=2) # 16 x 16 x 256
                    se9 = resBlock(se8, num_outputs=size * 16, kernel_size=4, stride=1) # 16 x 16 x 256
                    se10 = resBlock(se9, num_outputs=size * 32, kernel_size=4, stride=2) # 8 x 8 x 512
                    se11 = resBlock(se10, num_outputs=size * 32, kernel_size=4, stride=1) # 8 x 8 x 512

                    pd1 = tcl.conv2d_transpose(se11, size * 32, 4, stride=1) # 8 x 8 x 512 
                    pd2 = tcl.conv2d_transpose(pd1, size * 16, 4, stride=2) # 16 x 16 x 256 
                    pd3 = tcl.conv2d_transpose(pd2, size * 16, 4, stride=1) # 16 x 16 x 256 
                    pd4 = tcl.conv2d_transpose(pd3, size * 16, 4, stride=1) # 16 x 16 x 256 
                    pd5 = tcl.conv2d_transpose(pd4, size * 8, 4, stride=2) # 32 x 32 x 128 
                    pd6 = tcl.conv2d_transpose(pd5, size * 8, 4, stride=1) # 32 x 32 x 128 
                    pd7 = tcl.conv2d_transpose(pd6, size * 8, 4, stride=1) # 32 x 32 x 128 
                    pd8 = tcl.conv2d_transpose(pd7, size * 4, 4, stride=2) # 64 x 64 x 64 
                    pd9 = tcl.conv2d_transpose(pd8, size * 4, 4, stride=1) # 64 x 64 x 64 
                    pd10 = tcl.conv2d_transpose(pd9, size * 4, 4, stride=1) # 64 x 64 x 64 
                    
                    pd11 = tcl.conv2d_transpose(pd10, size * 2, 4, stride=2) # 128 x 128 x 32
                    pd12 = tcl.conv2d_transpose(pd11, size * 2, 4, stride=1) # 128 x 128 x 32
                    pd13 = tcl.conv2d_transpose(pd12, size, 4, stride=2) # 256 x 256 x 16
                    pd14 = tcl.conv2d_transpose(pd13, size, 4, stride=1) # 256 x 256 x 16

                    pd15 = tcl.conv2d_transpose(pd14, 3, 4, stride=1) # 256 x 256 x 3
                    pd16 = tcl.conv2d_transpose(pd15, 3, 4, stride=1) # 256 x 256 x 3
                    pos = tcl.conv2d_transpose(pd16, 3, 4, stride=1, activation_fn = tf.nn.sigmoid)#, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
                                
                    return pos, se11
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
    
    

class resfcn256_2(object):
    def __init__(self, resolution_inp = 256, resolution_op = 256, channel = 3, name = 'resfcn256_2'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training = True):
        with tf.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu, 
                                     normalizer_fn=tcl.batch_norm, 
                                     biases_initializer=None, 
                                     padding='SAME',
                                     weights_regularizer=tcl.l2_regularizer(0.0002)):
                    size = 16  
                    # x: s x s x 3
                    sea1 = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1) # 256 x 256 x 16
                    sea2 = resBlock(sea1, num_outputs=size * 2, kernel_size=4, stride=2) # 128 x 128 x 32
                    sea3 = resBlock(sea2, num_outputs=size * 2, kernel_size=4, stride=1) # 128 x 128 x 32
                    sea4 = resBlock(sea3, num_outputs=size * 4, kernel_size=4, stride=2) # 64 x 64 x 64
                    sea5 = resBlock(sea4, num_outputs=size * 4, kernel_size=4, stride=1) # 64 x 64 x 64
                    sea6 = resBlock(sea5, num_outputs=size * 8, kernel_size=4, stride=2) # 32 x 32 x 128
                    sea7 = resBlock(sea6, num_outputs=size * 8, kernel_size=4, stride=1) # 32 x 32 x 128
                    sea8 = resBlock(sea7, num_outputs=size * 16, kernel_size=4, stride=2) # 16 x 16 x 256
                    sea9 = resBlock(sea8, num_outputs=size * 16, kernel_size=4, stride=1) # 16 x 16 x 256
                    sea10 = resBlock(sea9, num_outputs=size * 32, kernel_size=4, stride=2) # 8 x 8 x 512
                    sea11 = resBlock(sea10, num_outputs=size * 32, kernel_size=4, stride=1) # 8 x 8 x 512

                    pda1 = tcl.conv2d_transpose(sea11, size * 32, 4, stride=1) # 8 x 8 x 512
                    pda2 = tcl.conv2d_transpose(pda1, size * 16, 4, stride=2) # 16 x 16 x 256
                    pda3 = tcl.conv2d_transpose(pda2, size * 16, 4, stride=1) # 16 x 16 x 256
                    pda4 = tcl.conv2d_transpose(pda3, size * 16, 4, stride=1) # 16 x 16 x 256
                    pda5 = tcl.conv2d_transpose(pda4, size * 8, 4, stride=2) # 32 x 32 x 128
                    pda6 = tcl.conv2d_transpose(pda5, size * 8, 4, stride=1) # 32 x 32 x 128
                    pda7 = tcl.conv2d_transpose(pda6, size * 8, 4, stride=1) # 32 x 32 x 128
                    pda8 = tcl.conv2d_transpose(pda7, size * 4, 4, stride=2) # 64 x 64 x 64
                    pda9 = tcl.conv2d_transpose(pda8, size * 4, 4, stride=1) # 64 x 64 x 64
                    pda10 = tcl.conv2d_transpose(pda9, size * 4, 4, stride=1) # 64 x 64 x 64
                    
                    pda11 = tcl.conv2d_transpose(pda10, size * 2, 4, stride=2) # 128 x 128 x 32
                    pda12 = tcl.conv2d_transpose(pda11, size * 2, 4, stride=1) # 128 x 128 x 32
                    pda13 = tcl.conv2d_transpose(pda12, size, 4, stride=2) # 256 x 256 x 16
                    pda14 = tcl.conv2d_transpose(pda13, size, 4, stride=1) # 256 x 256 x 16

                    pda15 = tcl.conv2d_transpose(pda14, 3, 4, stride=1) # 256 x 256 x 3
                    pda16 = tcl.conv2d_transpose(pda15, 3, 4, stride=1) # 256 x 256 x 3
                    posa = tcl.conv2d_transpose(pda16, 3, 4, stride=1, activation_fn = tf.nn.sigmoid)#, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

                    se1 = tcl.conv2d(posa, num_outputs=size, kernel_size=4, stride=1)  # 256 x 256 x 16
                    se2 = resBlock(se1, num_outputs=size * 2, kernel_size=4, stride=2)  # 128 x 128 x 32
                    se3 = resBlock(se2, num_outputs=size * 2, kernel_size=4, stride=1)  # 128 x 128 x 32
                    se4 = resBlock(se3, num_outputs=size * 4, kernel_size=4, stride=2)  # 64 x 64 x 64
                    se5 = resBlock(se4, num_outputs=size * 4, kernel_size=4, stride=1)  # 64 x 64 x 64
                    se6 = resBlock(se5, num_outputs=size * 8, kernel_size=4, stride=2)  # 32 x 32 x 128
                    se7 = resBlock(se6, num_outputs=size * 8, kernel_size=4, stride=1)  # 32 x 32 x 128
                    se8 = resBlock(se7, num_outputs=size * 16, kernel_size=4, stride=2)  # 16 x 16 x 256
                    se9 = resBlock(se8, num_outputs=size * 16, kernel_size=4, stride=1)  # 16 x 16 x 256
                    se10 = resBlock(se9, num_outputs=size * 32, kernel_size=4, stride=2)  # 8 x 8 x 512
                    se11 = resBlock(se10, num_outputs=size * 32, kernel_size=4, stride=1)  # 8 x 8 x 512

                    pd1 = tcl.conv2d_transpose(se11, size * 32, 4, stride=1)  # 8 x 8 x 512
                    pd2 = tcl.conv2d_transpose(pd1, size * 16, 4, stride=2)  # 16 x 16 x 256
                    pd3 = tcl.conv2d_transpose(pd2, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd4 = tcl.conv2d_transpose(pd3, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd5 = tcl.conv2d_transpose(pd4, size * 8, 4, stride=2)  # 32 x 32 x 128
                    pd6 = tcl.conv2d_transpose(pd5, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd7 = tcl.conv2d_transpose(pd6, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd8 = tcl.conv2d_transpose(pd7, size * 4, 4, stride=2)  # 64 x 64 x 64
                    pd9 = tcl.conv2d_transpose(pd8, size * 4, 4, stride=1)  # 64 x 64 x 64
                    pd10 = tcl.conv2d_transpose(pd9, size * 4, 4, stride=1)  # 64 x 64 x 64

                    pd11 = tcl.conv2d_transpose(pd10, size * 2, 4, stride=2)  # 128 x 128 x 32
                    pd12 = tcl.conv2d_transpose(pd11, size * 2, 4, stride=1)  # 128 x 128 x 32
                    pd13 = tcl.conv2d_transpose(pd12, size, 4, stride=2)  # 256 x 256 x 16
                    pd14 = tcl.conv2d_transpose(pd13, size, 4, stride=1)  # 256 x 256 x 16

                    pd15 = tcl.conv2d_transpose(pd14, 3, 4, stride=1)  # 256 x 256 x 3
                    pd16 = tcl.conv2d_transpose(pd15, 3, 4, stride=1)  # 256 x 256 x 3
                    pos = tcl.conv2d_transpose(pd16, 3, 4, stride=1, activation_fn=tf.nn.sigmoid)


                    return posa, pos
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class PosPrediction():
    def __init__(self, resolution_inp = 256, resolution_op = 256): 
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp*1.1

        # network type
        self.network= resfcn256_2(self.resolution_inp, self.resolution_op)

        # net forward
        self.x = tf.placeholder(tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])  
        self.x_op, _ = self.network(self.x, is_training = False)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    def restore(self, model_path):        
        tf.train.Saver(self.network.vars).restore(self.sess, model_path)
 
    def predict(self, image):
        pos = self.sess.run(self.x_op, 
                    feed_dict = {self.x: image[np.newaxis, :,:,:]})
        pos = np.squeeze(pos)
        return pos*self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op, 
                    feed_dict = {self.x: images})
        return pos*self.MaxPos


