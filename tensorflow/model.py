  # -*- coding:utf-8 -*-

import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform
import tensorflow as tf
import numpy as np

import config
import cifar10_input

FLAGS = tf.app.flags.FLAGS


class Vgg16Model(object):
    """VGG16用のモデル

    本来はBatch NormalizationはVgg16に含まれないため、flagでon/offできるようにする
    """
    def __init__(self, is_train, use_batchnorm=False):
        """コンストラクタ

        :param is_train:
        :param use_batchnorm: Trueならばすべてのconvolution後にBatchNormが入る
        """
        self.is_train = is_train
        self.use_batchnorm = use_batchnorm
        # See models/research/slim/nets/vgg.py
        self.wd = 0.0005
        self._model_variables = []

    def _variable_wd(self, name, shape, wd=None, initializer=None):
        """
        """
        if initializer is None:
            initializer = tf.contrib.layers.variance_scaling_initializer()

        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
        self._model_variables.append(var)

        if wd is not None:
            # TODO: 共有されている変数が呼び出し回数だけ作成されてしまう
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight")
            tf.add_to_collection('decays', weight_decay)

        return var

    def conv2d_relu(self, featmap, channel, filter_size=3, name=None):
        """Convolution2D + ReLUを定義する (NCHW前提)
        Poolingを含まないLayer定義を行う。設定によってはBatchNormalizationを行う

        :param tf.Tensor featmap: 入力
        :param int filter_size: filter size 正方形前提
        :param int channel: 出力チャンネル
        :rtype: tf.Tensor
        :return: 出力featmap
        """
        C = featmap.shape.as_list()[1]

        with tf.variable_scope(name):
            kernel = self._variable_wd("W", shape=[filter_size, filter_size, C, channel],
                                       wd=self.wd)
            featmap = tf.nn.conv2d(featmap, kernel, [1, 1, 1, 1], padding='SAME', data_format="NCHW")

            if self.use_batchnorm:
                featmap = tf.contrib.layers.batch_norm(
                    featmap, center=True, scale=False,
                    is_training=self.is_train, data_format="NCHW", fused=True)
            else:
                biases = self._variable_wd('b', shape=[channel], wd=0.0, initializer=tf.constant_initializer())
                featmap = tf.nn.bias_add(featmap, biases, data_format="NCHW")

            featmap = tf.nn.relu(featmap, name="relu")

        return featmap

    def fc(self, features, channel, name=None):
        """このネットワーク内で用いるFC層

        :param tf.Tensor features: 入力 [NC]
        :param int channel: 出力チャンネル
        :param name:
        :return:
        """
        C = features.shape.as_list()[1]

        with tf.variable_scope(name):
            W = self._variable_wd("W", shape=[C, channel],wd=self.wd)
            b = self._variable_wd("b", shape=[channel],wd=0)
            features = tf.matmul(features, W) + b
        return features


    def max_pool(self, featmap, name):
        return tf.contrib.layers.max_pool2d(featmap, kernel_size=[2, 2], stride=[2, 2],
                              padding='SAME', data_format="NCHW")

    def inference(self, images):
        """
        :param tf.Tensor images: NHWC
        :rtype: tf.Tensor
        :return: logits
        """
        with tf.variable_scope("network"):
            logits = self._inference(images)

        return logits

    def _inference(self, images):
        """推論の本体
        VGG16の論文のDを元にしているが画像サイズが小さいためいくつかのmax-poolをスキップしている
        また、FC layerの計算量も軽いことに注意
        See https://arxiv.org/pdf/1409.1556.pdf

        :param tf.Tensor images: NHWC
        :rtype: tf.Tensor
        :return: logits
        """
        # NHWC => NCHW
        featmap = tf.transpose(images, [0,3,1,2])

        featmap = self.conv2d_relu(featmap, 64, name="conv1")
        featmap = self.conv2d_relu(featmap, 64, name="conv2")

        # MEMO: 元論文では入っているが画像サイズが小さいためスキップ
        # featmap = self.max_pool(featmap, name="pool1")

        featmap = self.conv2d_relu(featmap, 128, name="conv3")
        featmap = self.conv2d_relu(featmap, 128, name="conv4")
        # MEMO: 元論文では入っているが画像サイズが小さいためスキップ
        # featmap = self.max_pool(featmap, name="pool2")

        featmap = self.conv2d_relu(featmap, 256, name="conv5")
        featmap = self.conv2d_relu(featmap, 256, name="conv6")
        featmap = self.conv2d_relu(featmap, 256, name="conv7")

        featmap = self.max_pool(featmap, name="pool3")

        featmap = self.conv2d_relu(featmap, 512, name="conv8")
        featmap = self.conv2d_relu(featmap, 512, name="conv9")
        featmap = self.conv2d_relu(featmap, 512, name="conv10")

        featmap = self.max_pool(featmap, name="pool4")

        featmap = self.conv2d_relu(featmap, 512, name="conv11")
        featmap = self.conv2d_relu(featmap, 512, name="conv12")
        featmap = self.conv2d_relu(featmap, 512, name="conv13")

        featmap = self.max_pool(featmap, name="pool5")

        features = tf.contrib.layers.flatten(featmap)

        features = self.fc(features, channel=4096, name="fc1")
        features = tf.nn.relu(features)
        features = tf.layers.dropout(features, rate=0.5, training=self.is_train)

        features = self.fc(features, channel=4096, name="fc2")
        features = tf.nn.relu(features)
        features = tf.layers.dropout(features, rate=0.5, training=self.is_train)

        logits = self.fc(features, channel=10, name="fc3")

        return logits

    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        decays = tf.get_collection('decays')
        decay_loss = tf.add_n(decays, 'decay_loss')
        total_loss = tf.add(cross_entropy_mean, decay_loss, name="total_loss")

        tf.add_to_collection("view_losses", cross_entropy_mean)
        tf.add_to_collection("view_losses", decay_loss)
        tf.add_to_collection("view_losses", total_loss)

        return total_loss

