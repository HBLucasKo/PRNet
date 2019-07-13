

import numpy as np
import os
import argparse
import tensorflow as tf
import cv2
import random

from scipy.ndimage import imread

random.seed(0)
from predictor_2 import resfcn256
from predictor_2 import resfcn256_2
import math
import imageio
import matplotlib.pyplot as plt


class TrainData(object):

    def __init__(self, train_data_file):
        super(TrainData, self).__init__()
        self.train_data_file = train_data_file
        self.train_data_list = []
        self.readTrainData()
        self.index = 0
        self.num_data = len(self.train_data_list)

    def readTrainData(self):
        with open(self.train_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip().split()
                self.train_data_list.append(item)
            random.shuffle(self.train_data_list)

    def getBatch(self, batch_list):
        batch = []
        imgs = []
        labels = []
        labels_texture = []
        for item in batch_list:
            img = imread(item[0])
            label = np.load(item[1])
            #label_tex = imread(item[2])

            im_array = np.array(img, dtype=np.float32)
            imgs.append(im_array / 255.0)

            label_array = np.array(label, dtype=np.float32)
            labels_array_norm = (label_array - label_array.min()) / (label_array.max() - label_array.min())
            labels.append(labels_array_norm)

        batch.append(imgs)
        batch.append(labels)

        return batch

    def __call__(self, batch_num):
        if (self.index + batch_num) <= self.num_data:
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num

            return batch_data
        elif self.index < self.num_data:
            batch_list = self.train_data_list[self.index:self.num_data]
            batch_data = self.getBatch(batch_list)
            self.index = 0
            return batch_data
        else:
            self.index = 0
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num
            return batch_data


def main(args):
    # Some arguments
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    train_data_file = args.train_data_file
    learning_rate = args.learning_rate
    model_path = args.model_path

    save_dir = args.checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Training data
    data = TrainData(train_data_file)

    x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    label = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    label_texture = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])

    # Train net

    #net1 = resfcn256(256, 256)
    net2 = resfcn256_2(256,256)

    # sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.import_meta_graph(r'C:\Users\CVPR\PRNet\checkpoint\256_256_resfcn256_weight.meta')
    # saver.restore(sess, r'C:\Users\CVPR\PRNet\checkpoint\256_256_resfcn256_weight')
    # net = saver
    x_op, x_fin = net2(x, is_training=True)
    # x_op, _ = net1(x, is_training=True)
    # x_fin = net2(x, is_training=True)
    tf.summary.image('x_op', x_op)
    weights = imageio.imread('C:\\Users\\CVPR\\PRNet\\Data\\uv-data\\uv_weight_mask.png')
    #weights = weights.astype(np.float32) / 255.0
    weights = weights.reshape(1, weights.shape[0], weights.shape[1], 1)
    weights = tf.constant(weights)
    # weights = tf.broadcast_to(weights, [1, 256,256,3])
    # Loss
    loss_1 = tf.losses.mean_squared_error(label, x_op, weights=weights)
    loss_2 = tf.losses.mean_squared_error(x, x_fin)
    loss = loss_1 + (0.01)*loss_2
    learning_rate = tf.train.cosine_decay_restarts(learning_rate=learning_rate, global_step=100,
                                                   first_decay_steps=10000)

    # This is for batch norm layer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(loss)
    # train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    tf.summary.image('label', label)
    tf.summary.scalar('loss', loss)
    tf.summary.image('x_op', x_op)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/hello_tf_190704-1")
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    #if os.path.exists('./checkpoint'):
        #tf.train.Saver(net1.vars).restore(sess, model_path)
       # print("restoring")

    saver = tf.train.Saver(var_list=tf.global_variables())
    save_path = model_path

    # Begining train
    fig1 = plt.figure()
    #fig2 = plt.figure()
    for epoch in range(epochs):
        for i in range(int(math.ceil(1.0 * data.num_data / batch_size))):
            batch = data(batch_size)
            # loss_res = sess.run(loss,feed_dict={label:batch[1], x:batch[0]})
            _, loss_res, uv_rec, fin_rec = sess.run([train_step, loss, x_op, x_fin], feed_dict={x: batch[0], label: batch[1]})

            print('epoch:%d,loss:%f' % (epoch, loss_res))


            if i % 1000 == 0:
                fig1.clf()
                plt.imshow(uv_rec[0])
                #fig2.clf()
                #plt.imshow(fin_rec[0])
                plt.pause(0.0001)

        saver.save(sess=sess, save_path=save_path)
    plt.show()


if __name__ == '__main__':
    par = argparse.ArgumentParser(
        description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    par.add_argument('--train_data_file', default='Data/trainData/trainDataLabel.txt', type=str,
                     help='The training data file')
    par.add_argument('--learning_rate', default=0.001, type=float, help='The learning rate')
    par.add_argument('--epochs', default=20, type=int, help='Total epochs')
    par.add_argument('--batch_size', default=32, type=int, help='Batch sizes')
    par.add_argument('--checkpoint', default='checkpoint/', type=str, help='The path of checkpoint')
    par.add_argument('--model_path', default='checkpoint/256_256_resfcn256_weight', type=str,
                     help='The path of pretrained model')
    par.add_argument('--gpu', default='0', type=str, help='The GPU ID')

    main(par.parse_args())



