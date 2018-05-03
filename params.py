from math import ceil,sqrt,log2

class GeneralParameters():
    def __init__(self):
        # seed for randomizing numpy and TensorFlow
        self.seed = 8734

class DataParams():
    def __init__(self):
        self.num_classes = None
        self.number_of_channels = None
        self.image_size = None

        self.training_set_size = None
        self.validation_set_size = None
        self.test_set_size = None

        self.path = None

class ImageNetParams(DataParams):
    def __init__(self):
        self.num_classes = 1000
        self.number_of_channels = 3
        self.image_size = 224

        self.training_set_size = 1281167
        self.validation_set_size = 50000
        self.test_set_size = 1

        self.path = 'data/imagenet/'

class Cifar10Params(DataParams):
    def __init__(self):
        self.num_classes = 10
        self.number_of_channels = 3
        self.image_size = 32

        self.training_set_size = 50000
        self.validation_set_size = 10000
        self.test_set_size = 0

        self.path = 'data/cifar10/'



class ResNetHyperParameters():
    def __init__(self):
        self.first_layer_filter_size = None
        self.first_layer_size = None
        self.first_layer_pool_size = None
        self.filter_size = None
        self.stride_between_scales = None
        self.layers_per_block = None
        self.num_blocks_per_scale = None

        self.momentum = None
        self.learning_rate = None

class ResNetImageNetHyperParameters(ResNetHyperParameters):
    def __init__(self):
        self.first_layer_filter_size = 7
        self.first_layer_size = 64
        self.first_layer_pool_size = 3
        self.filter_size = 3
        self.stride_between_scales = 2
        self.layers_per_block = 2
        self.num_blocks_per_scale = [3, 4, 6, 3]

        self.momentum = 0.9
        self.learning_rate = 0.1

class ResNetCifar10HyperParameters(ResNetHyperParameters):
    def __init__(self):
        self.first_layer_filter_size = 3
        self.first_layer_size = 16
        self.first_layer_pool_size = 3
        self.filter_size = 3
        self.stride_between_scales = 2
        self.layers_per_block = 2
        self.num_blocks_per_scale = [5,5,5]

        self.momentum = 0.9
        self.learning_rate = 0.1


class HypernetworkHyperParameters():
    """
    hyperparameters for hypernetwork (=generator)
    """
    def __init__(self):
        self.with_batchnorm = True # should use batch normalization?
        self.batchnorm_decay = 0.98 # exponential decay constant for batch normalization

        self.input_noise_size = 500  # dimension of noise vector z
        self.input_noise_bound = 1  # z will be sampled from a uniform distribution [-input_noise_bound,input_noise_bound]
        self.e_layer_sizes = [800,800,800] # sizes of hidden layers for extractor
        self.wg_number_of_hidden_layers = 3 # number of hidden layers for each weight generator
        self.wg_hidden_layer_size_formula = lambda filter_size, number_of_filters: ceil(12 * sqrt(filter_size)) # size of each hidden layer for weight generator
        self.code_size_formula = lambda filter_size,number_of_filters: ceil(log2(filter_size)) # code size at the input the weight generator

        self.zero_fixer = 1e-8 # add this constant to the argument of sqrt, log, etc. so that the argument is never zero

        self.initialization_std = 1e-1 # used for weights and biases
        self.batch_size = 50 # batch size, during training
        self.validation_samples = 200  # how many times to sample the validation loss for getting an estimate of mean validation loss
        self.learning_rate = 3e-4  # initial learning rate
        self.learning_rate_rate = 0.99998  # decay rate of learning rate - decay happens once every training step
        self.momentum = 0.9
        self.lamBda = 1e3  #  lambda value (=coefficient of accuracy component in total loss)
