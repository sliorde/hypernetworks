import os

import tensorflow as tf
import numpy as np

from utils import CreateOutputDir, GetLogger

from hypernetwork import Hypernetwork
from cifar10_data_fetcher import Cifar10DataFetcher
from params import GeneralParameters, HypernetworkHyperParameters

initialize_from_checkpoint = False
output_dir = CreateOutputDir('output/', __file__)
logger = GetLogger(log_file_mode='a' if initialize_from_checkpoint else 'w', log_file_path=os.path.join(output_dir, 'log.txt'))

general_params = GeneralParameters()
hparams = HypernetworkHyperParameters()

np.random.seed(general_params.seed)
tf.set_random_seed(general_params.seed)

training_data = Cifar10DataFetcher('TRAIN', batch_size=hparams.batch_size, order='NHWC')

hnet = Hypernetwork(training_data.image, training_data.label, 'TRAIN', hnet_hparams=hparams)
with tf.Session() as sess:
    hnet.Train(sess, max_steps=1e6, logger=logger, checkpoint_file_name=output_dir)
