# experiment:
# - as (6), lower decay, residual_connections

import os

import tensorflow as tf
import numpy as np

from utils import CreateOutputDir, GetLogger

from hypernetwork import Hypernetwork
from cifar10_data_fetcher import Cifar10DataFetcher
from params import GeneralParameters, HypernetworkHyperParameters, Cifar10Params, ResNetCifar10HyperParameters

initialize_from_checkpoint = False
output_dir = CreateOutputDir('output/', __file__)
logger = GetLogger(log_file_mode='a' if initialize_from_checkpoint else 'w', log_file_path=os.path.join(output_dir, 'log.txt'))

# create parameter objects
general_params = GeneralParameters()
hparams = HypernetworkHyperParameters()
target_hparams = ResNetCifar10HyperParameters()
image_params = Cifar10Params()

# override
hparams.initialization_std = 1e-3
hparams.learning_rate = 4e-7
hparams.learning_rate_rate = 0.99999
hparams.create_optimizer = lambda learning_rate, hnet_hparams: tf.train.AdamOptimizer(learning_rate)
hparams.with_residual_connections = True

target_hparams.batch_type = 'BATCH_TYPE5'

# set random seeds
np.random.seed(general_params.seed)
tf.set_random_seed(general_params.seed)

config = tf.ConfigProto(allow_soft_placement=True)
config.log_device_placement = False
config.gpu_options.per_process_gpu_memory_fraction = 0.5
with tf.Session(config=config).as_default() as sess:

    # get training data pipeline.
    training_data = Cifar10DataFetcher('TRAIN', batch_size=hparams.batch_size)
    validation_data = Cifar10DataFetcher('VALIDATION', batch_size=hparams.batch_size)

    # create hypernet
    hnet = Hypernetwork(training_data.image, training_data.label, 'TRAIN', hnet_hparams=hparams)

    # protobuf computational graph
    with open(os.path.join(output_dir, 'graph.pb'), 'w') as f:
        f.write(str(hnet.graph.as_graph_def()))

    # summary
    writer = tf.summary.FileWriter(output_dir, hnet.graph)

    # train
    hnet.Train(sess, validation_data.image, validation_data.label, max_steps=1e6, logger=logger, writer=writer, checkpoint_file_name=output_dir, log_interval=100)
