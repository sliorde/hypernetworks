import os

import tensorflow as tf
import numpy as np
from time import time

from utils import CreateOutputDir, MultiLayerPerceptron, MultiLayerPerceptron2

from params import GeneralParameters, HypernetworkHyperParameters, ResNetCifar10HyperParameters, Cifar10Params

from target_network import Resnet
from target_network_2 import Resnet2

from cifar10_data_fetcher import Cifar10DataFetcher


output_dir = CreateOutputDir('output/', __file__)

general_params = GeneralParameters()
hnet_hparams = HypernetworkHyperParameters()
target_hparams = ResNetCifar10HyperParameters()
image_params = Cifar10Params()

np.random.seed(general_params.seed)
tf.set_random_seed(general_params.seed)


# original
tf.reset_default_graph()
graph = tf.get_default_graph()

training_data = Cifar10DataFetcher('TRAIN', batch_size=hnet_hparams.batch_size, order='NHWC') # TODO should be NCHW
resnet = Resnet(input=training_data.image, hparams=target_hparams, image_params=image_params, weights=None, order='NHWC', graph=graph) # TODO should be NCHW
init = tf.global_variables_initializer()

graph.finalize()

with open(os.path.join(output_dir, 'graph_1.pb'), 'w') as f:
    f.write(str(graph.as_graph_def()))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    assert len(graph.get_operations()) == 2896
    assert len(tf.all_variables()) == 101

    sess.run(init)

    t = time()
    for i in range(1000):
        sess.run(resnet.predictions)
    t1 = time() - t



# modified
tf.reset_default_graph()
graph = tf.get_default_graph()

training_data = Cifar10DataFetcher('TRAIN', batch_size=hnet_hparams.batch_size, order='NHWC')  # TODO should be NCHW
resnet = Resnet2(input=training_data.image, hparams=target_hparams, image_params=image_params, weights=None, order='NHWC', graph=graph)  # TODO should be NCHW
init = tf.global_variables_initializer()

graph.finalize()

with open(os.path.join(output_dir, 'graph_2.pb'), 'w') as f:
    f.write(str(graph.as_graph_def()))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    assert len(graph.get_operations()) == 2896
    assert len(tf.all_variables()) == 101

    sess.run(init)

    t = time()
    for i in range(1000):
        sess.run(resnet.predictions)
    t2 = time() - t

print(t1)
print(t2)
print('done.')
