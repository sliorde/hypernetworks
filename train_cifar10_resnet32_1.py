import os

import tensorflow as tf
import numpy as np

from utils import CreateOutputDir, GetLogger

from hypernetwork import Hypernetwork
from cifar10_data_fetcher import Cifar10DataFetcher
from params import GeneralParameters, HypernetworkHyperParameters

initialize_from_checkpoint = False
output_dir = CreateOutputDir('output', __file__)
logger = GetLogger(log_file_mode='a' if initialize_from_checkpoint else 'w', log_file_path=os.path.join(output_dir, 'log.txt'))

# create parameter objects
general_params = GeneralParameters()
hparams = HypernetworkHyperParameters()

# set random seeds
np.random.seed(general_params.seed)
tf.set_random_seed(general_params.seed)

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
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    hnet.Train(sess,validation_data.image,validation_data.label, max_steps=1e6, logger=logger, writer=writer, checkpoint_file_name=output_dir)