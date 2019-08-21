#-*- coding:utf-8 -*-

import os

import tensorflow.python.platform
import tensorflow as tf

from tensorflow.python.platform import gfile
import model
import glob
import logging
import fnmatch

FLAGS = tf.app.flags.FLAGS

class QueueInput(object):
    """画像のTensorを生成するクラス

    :param boolean is_train: 学習モードかどうか
    :param str dir_data: data_batch_1.bin ... を含むディレクトリのパス
    """
    def __init__(self, is_train, dir_data):
        self.is_train = is_train
        self.list_input = self._enumerate_target(dir_data)

    def _enumerate_target(self, dir_data):
        """今回対象となるファイル群のフルパスを返す


        :return:
        """
        logger = logging.getLogger(__name__)

        train_names = [
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin"]
        test_names = ["test_batch.bin"]

        ###Python3.5以上ならこれでもいけるはず↲
        #train_names = glob.glob('**/data_batch_*.bin', recursive=True)
        #test_names = glob.glob('**/test_batch_*.bin', recursive=True)

        train_names = []
        test_names = []
        for root, dir, files in os.walk(dir_data):
            for fname in files:
                if fnmatch.fnmatch(fname, 'data_batch_*.bin'):
                    train_names.append(os.path.join(root, fname))
                elif fnmatch.fnmatch(fname, 'test_batch.bin'):
                    test_names.append(os.path.join(root, fname))

        if self.is_train:
            names = train_names
        else:
            names = test_names

        list_path = []
        for cur_name in names:
            fullpath = cur_name   #os.path.join(dir_data, cur_name)
            if not os.path.exists(fullpath):
                raise ValueError("{} is not found.".format(fullpath))
            list_path.append(fullpath)

        logger.debug("target files : {}".format(list_path))
        return list_path

    def preprocess(self, image):
        """画像に前処理を行う。学習・推論共通

        :param tf.Tensor image: [NHWC]のuint8であるTensor
        :rtype: tf.Tensor
        :return: preprocessed image
        """
        image = tf.cast(image, tf.float32)
        image = image / 255

        return image

    def _augument(self, image):
        """画像をAugmentする。学習時のみ
        現在は何もしない

        :param tf.Tensor: NHWC float32
        :rtype: tf.Tensor
        :return: augmented image
        """
        return image

    def get_batch(self, batch_size):
        """入力バッチを取得する

        :param int batch_size: batch_size
        :return:
        """
        # Create a queue that produces the filenames to read.
        if self.is_train:
            filename_queue = tf.train.string_input_producer(self.list_input)
        else:
            filename_queue = tf.train.string_input_producer(self.list_input, num_epochs=1)

        image, label = self.read_cifar10(filename_queue)
        image = self.preprocess(image)

        if self.is_train:
            image = self._augument(image)
            #min_queue_examples = 256
            min_queue_examples = batch_size * 2

            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=FLAGS.threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)

        else:
            batch_size = FLAGS.batch_size
            min_queue_examples = 256

            images, labels = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=FLAGS.threads,
                capacity=min_queue_examples + 3 * batch_size,
                allow_smaller_final_batch=True
            )

        return images, tf.reshape(labels, [-1])


    def read_cifar10(self, filename_queue):
        """ファイル名のQueueを受け取り画像とlabelのTensorペアを返す

        :param tf.Queue filename_queue: ファイル名を順次返すQueue
        """
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        label_bytes = 1
        height, width, channel = 32, 32, 3
        image_bytes = height*width*channel
        record_bytes = label_bytes + image_bytes

        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        label = tf.cast(
                tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                                                         [channel, height, width])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.transpose(depth_major, [1, 2, 0])

        return image, label


