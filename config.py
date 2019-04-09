#-*- coding:utf-8 -*-

"""
設定項目をまとめたファイル
"""

import tensorflow as tf


# 出力データパラメーター #######
tf.app.flags.DEFINE_string('parameter_dir', './output/parameter',
                           """Directory where to write parameters """)

tf.app.flags.DEFINE_string('train_log_dir', './output/train_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_log_dir', './output/eval_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# 入力データパラメーター ########
tf.app.flags.DEFINE_string('images', './Image/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")

# 学習パラメーター
tf.app.flags.DEFINE_integer('max_steps', 90000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('checkpoint', None,
                            """Checkpoint file restored""")

tf.app.flags.DEFINE_integer('threads', 8,
                            """Number of input threads.""")

tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_float('lr', 0.001,
                          """initial learning rate.""")

tf.app.flags.DEFINE_integer('decay_steps', 20000,
                            """decay steps""")

LEARNING_RATE_DECAY_FACTOR = 0.1    # Learning rate decay factor.

def get_config_str():
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    return pp.pformat(tf.app.flags.FLAGS.__flags)
