#!/usr/bin/python
# -*- coding:utf-8 -*-

"""Evaluation for CIFAR-10.

"""
from datetime import datetime
import sys
import time
import logging

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import config
import cifar10_input
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('eval_interval_secs', 60*5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('oneshot', False,
                            """eval only one time.""")


def restore_model(sess, saver, dir_ckpt):
    """
    このモデルの復元に必要な変数のみrestoreする
    :return:
      global_step : 復元が失敗したらNone
    """
    logger = logging.getLogger(__name__)
    ckpt_state = tf.train.get_checkpoint_state(dir_ckpt)

    if ckpt_state is None or ckpt_state.model_checkpoint_path is None:
        logger.warn("checkpoint is not found in {}.".format(dir_ckpt))
        return None

    path_ckpt = ckpt_state.model_checkpoint_path
    global_step = path_ckpt.split('/')[-1].split('-')[-1]

    saver.restore(sess, path_ckpt)

    return global_step

def eval_once(summary_writer, labels, predicts):
    """一度精度評価を行う。

    :param tf.Tensor labels: 正解ラベル
    :param tf.Tensor predicts: 予測ラベル
    """
    logger = logging.getLogger(__name__)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    saver = tf.train.Saver()

    with tf.train.SingularMonitoredSession(config=sess_config) as sess:
        # とりあえず仮で全変数をrestore
        global_step = restore_model(sess, saver, FLAGS.parameter_dir)

        all_labels = []
        all_preds = []

        try:
            while not sess.should_stop():
                cur_labels, cur_preds = sess.run([labels, predicts])

                all_labels.extend(cur_labels.tolist())
                all_preds.extend(cur_preds.tolist())

        except tf.errors.OutOfRangeError as e:
            #logger.warn("Out of range error", exc_info=True)
            pass

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        true_counts = np.sum(all_labels == all_preds)

        precision = true_counts / float(len(all_labels))
        print ("Data Count : {}".format(len(all_labels)))
        print('%s : precision @ 1 = %.3f' % (datetime.now(), precision))

        summary = tf.Summary()
        summary.value.add(tag='Precision/Precision', simple_value=precision)

        summary_writer.add_summary(summary, global_step)
        summary_writer.flush()


def evaluate():
    with tf.Graph().as_default() as g:
        network = model.Vgg16Model(is_train=False)
        feeder = cifar10_input.QueueInput(is_train=False, dir_data=FLAGS.images)
        images, labels = feeder.get_batch(FLAGS.batch_size)

        logits = network.inference(images)
        predicts = tf.argmax(logits, dimension=1)

        summary_writer = tf.summary.FileWriter(FLAGS.eval_log_dir, g)

        while True:
            eval_once(summary_writer, labels, predicts)

            if FLAGS.oneshot:
                break

            time.sleep(FLAGS.eval_interval_secs)        


def main(argv=None):    # pylint: disable=unused-argument
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )

    if not gfile.Exists(FLAGS.parameter_dir):
        print ("parameter_dir {} not exists.".format(FLAGS.parameter_dir))
        sys.exit(-1)

    if gfile.Exists(FLAGS.eval_log_dir):
        gfile.DeleteRecursively(FLAGS.eval_log_dir)

    gfile.MakeDirs(FLAGS.eval_log_dir)

    print(config.get_config_str())
    
    evaluate()


if __name__ == '__main__':
    tf.app.run()
