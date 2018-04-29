import tensorflow as tf
import numpy as np
from hypernetwork import Hypernetwork
from cifar10_data_fetcher import Cifar10DataFetcher
from params import GeneralParameters, HypernetworkHyperParameters

general_params = GeneralParameters()
hparams = HypernetworkHyperParameters()

np.random.seed(general_params.seed)
tf.set_random_seed(general_params.seed)

training_data = Cifar10DataFetcher('TRAIN',batch_size=hparams.batch_size,order='NHWC')

hnet = Hypernetwork(training_data.image,training_data.label,'TRAIN',hnet_hparams=hparams)
with tf.Session() as sess:
    hnet.Train(sess,1e6,checkpoint_file_name='temp/')