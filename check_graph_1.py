import os

import tensorflow as tf
import numpy as np
from time import time

from utils import CreateOutputDir, MultiLayerPerceptron, MultiLayerPerceptron2

from params import GeneralParameters, HypernetworkHyperParameters


output_dir = CreateOutputDir('output/', __file__)

general_params = GeneralParameters()
hnet_hparams = HypernetworkHyperParameters()

np.random.seed(general_params.seed)
tf.set_random_seed(general_params.seed)


# with lambda
tf.reset_default_graph()
graph = tf.get_default_graph()
mlp_builder = lambda input, widths, name=None: MultiLayerPerceptron(input, widths, with_batch_norm=hnet_hparams.with_batchnorm, scale=np.square(hnet_hparams.initialization_std), batchnorm_decay=hnet_hparams.batchnorm_decay, is_training=True, name=name, zero_fixer=hnet_hparams.zero_fixer)[0]

z = tf.placeholder(tf.float32, [None, hnet_hparams.input_noise_size])
e_layer_outputs = mlp_builder(z, hnet_hparams.e_layer_sizes, 'extractor')

with open(os.path.join(output_dir, 'graph_1.pb'), 'w') as f:
    f.write(str(graph.as_graph_def()))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    assert len(graph.get_operations()) == 268
    assert len(tf.all_variables()) == 12

    sess.run(tf.global_variables_initializer())

    t = time()
    for i in range(10000):
        z_val = np.random.uniform(-1*hnet_hparams.input_noise_bound,hnet_hparams.input_noise_bound,[1,hnet_hparams.input_noise_size])
        sess.run(e_layer_outputs, feed_dict={z: z_val})
    print(time() - t)


# without lambda
tf.reset_default_graph()
graph = tf.get_default_graph()


z = tf.placeholder(tf.float32, [None, hnet_hparams.input_noise_size])
e_layer_outputs = MultiLayerPerceptron2(z, hnet_hparams.e_layer_sizes, with_batch_norm=hnet_hparams.with_batchnorm, scale=np.square(hnet_hparams.initialization_std), batchnorm_decay=hnet_hparams.batchnorm_decay, is_training=True, name='extractor2', zero_fixer=hnet_hparams.zero_fixer)[0]

with open(os.path.join(output_dir, 'graph_2.pb'), 'w') as f:
    f.write(str(graph.as_graph_def()))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    assert len(graph.get_operations()) == 268
    assert len(tf.all_variables()) == 12

    sess.run(tf.global_variables_initializer())

    t = time()
    for i in range(10000):
        z_val = np.random.uniform(-1*hnet_hparams.input_noise_bound,hnet_hparams.input_noise_bound,[1,hnet_hparams.input_noise_size])
        sess.run(e_layer_outputs, feed_dict={z: z_val})
    print(time() - t)

print('done.')
