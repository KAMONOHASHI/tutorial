#!/usr/bin/python
#-*- coding:utf-8 -*-

import time
import logging

import tensorflow as tf
from tensorflow.python.platform import gfile

import config
import model

from cifar10_input import QueueInput

FLAGS = tf.app.flags.FLAGS


def _add_loss_summaries():
    """Tensorboard用に損失関数のtrackingを追加する

    view_losses collectionに含まれる損失をトラックする

    :rtype: tf.Operator
    :return:
        loss_averages_op: op for generating moving averages of losses.
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('view_losses')
    loss_averages_op = loss_averages.apply(losses)

    for l in losses:
        tf.summary.scalar(l.op.name +'_raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    lr = tf.train.exponential_decay(FLAGS.lr,
                                    global_step,
                                    FLAGS.decay_steps,
                                    config.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    updates = tf.group(*update_ops)

    optimizer = tf.train.AdamOptimizer(lr)
    minimizer = optimizer.minimize(total_loss, global_step=global_step)

    train_op = tf.group(updates, loss_averages_op, minimizer, name="train")

    return train_op


def start_train():
    logger = logging.getLogger(__name__)

    with tf.Graph().as_default():
        global_step = tf.train.create_global_step()

        feeder = QueueInput(is_train=True, dir_data=FLAGS.images)
        network = model.Vgg16Model(is_train=True, use_batchnorm=False)

        images, labels = feeder.get_batch(batch_size=FLAGS.batch_size)
        logits = network.inference(images)
        loss = network.loss(logits, labels)

        train_op = train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        summary_op = tf.summary.merge_all()        

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
        session_config = tf.ConfigProto(gpu_options=gpu_options)

        nan_hook = tf.train.NanTensorHook(loss_tensor=loss)
        saver_hook = tf.train.CheckpointSaverHook(saver=saver, checkpoint_dir=FLAGS.parameter_dir, save_steps=1000)
        summary_hook = tf.train.SummarySaverHook(output_dir=FLAGS.train_log_dir, summary_op=summary_op,
                                                 save_steps=500)

        hooks = [nan_hook, summary_hook, saver_hook, tf.train.StopAtStepHook(last_step=FLAGS.max_steps)]

        with tf.train.SingularMonitoredSession(config=session_config, hooks=hooks) as sess:
            num_iter = 0
            while not sess.should_stop():
                start_time = time.time()

                cur_loss, _ = sess.run([loss, train_op])

                duration = time.time() - start_time

                if num_iter % 100 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration

                    sec_per_batch = float(duration)

                    logger.info("step = {} loss = {:.2f} ({:.1f} examples/sec; {:.1f} sec/batch)"
                                .format(num_iter, cur_loss, examples_per_sec, sec_per_batch))

                num_iter += 1

def main(argv=None):
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )

    if gfile.Exists(FLAGS.parameter_dir):
        gfile.DeleteRecursively(FLAGS.parameter_dir)
    if gfile.Exists(FLAGS.train_log_dir):
        gfile.DeleteRecursively(FLAGS.train_log_dir)

    gfile.MakeDirs(FLAGS.parameter_dir)
    gfile.MakeDirs(FLAGS.train_log_dir)

    print(config.get_config_str())    
    
    start_train()


if __name__ == '__main__':
    tf.app.run()
