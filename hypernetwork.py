import numpy as np
import tensorflow as tf
from itertools import cycle
from logging import Logger
import os

from utils import MultiLayerPerceptron, OptimizerReset,GetDevices,GetLogger
from params import GeneralParameters, HypernetworkHyperParameters, ResNetHyperParameters, DataParams, Cifar10Params,ResNetCifar10HyperParameters
from target_network import Resnet, ResnetWeights

CHECKPOINT_FILENAME = './checkpoints/checkpoint'
INITIALIZE_FROM_CHECKPOINT = False
CHECKPOINT_STEP = 0
CHECKPOINT_MESSAGE = None

LEARNING_RATE_FILE_NAME = 'lr.txt'
LEARNING_RATE_RATE_FILE_NAME = 'lrr.txt'
update_dict = {'learning_rate':LEARNING_RATE_FILE_NAME,'learning_rate_rate':LEARNING_RATE_RATE_FILE_NAME}

class Hypernetwork():
    def __init__(self, x,y, mode:str='EVAL', general_params=GeneralParameters(), hnet_hparams=HypernetworkHyperParameters(), target_hparams:ResNetHyperParameters=ResNetCifar10HyperParameters(), image_params:DataParams=Cifar10Params(), devices=None,graph=None):

        if mode.upper() in ['TRAIN','TRAINING']:
            mode = 'TRAIN'
        elif mode.upper() in ['VALIDATE', 'VALIDATION', 'VALIDATING', 'EVALUATE', 'EVAL', 'EVALUATION', 'EVALUATING', 'TEST', 'TESTING']:
            mode = 'EVAL'
        else:
            raise ValueError("mode needs to be one of 'TRAIN','VALIDATE','INFER'")

        self.general_params = general_params
        self.hnet_hparams = hnet_hparams
        self.target_hparams = target_hparams
        self.image_params = image_params
        self.graph = graph

        if devices is None:
            devices = GetDevices()
        self.cpu = devices['cpu']
        self.gpus = cycle(devices['gpus']) if len(devices['gpus']) > 0 else cycle([devices['cpu']])

        self.x = x
        self.y = y

        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph

        with self.graph.as_default():
            self.z = tf.placeholder(tf.float32,[None,hnet_hparams.input_noise_size])
            # self.z = tf.random_uniform([hnet_hparams.batch_size,hnet_hparams.input_noise_size],-1*hnet_hparams.input_noise_bound,hnet_hparams.input_noise_bound)
            self.is_training = tf.Variable(False, trainable=False, name='is_training')
            self.__AddGeneratorOps()
            if mode == 'TRAIN':
                self.__AddOptimizationOps()
            self.initializer = tf.variables_initializer(self.graph.get_collection('variables'),name='initializer')


    def __AddGeneratorOps(self):

        batch_size = tf.shape(self.z)[0]
        # batch_size = self.hnet_hparams.batch_size

        weights = ResnetWeights(self.target_hparams,self.image_params)
        layers = list(weights.WeightedLayerIterator())
        num_of_weights_per_filter = {}
        num_of_filters = {}
        for l in layers:
            num_of_weights_per_filter[l] = np.prod(l.w_shape[:(-1)])
            num_of_weights_per_filter[l] += 1 if l.b_shape is not None else 0
            num_of_filters[l] = l.w_shape[-1]

        with tf.device(self.cpu):
            step_counter = tf.Variable(0, trainable=False,name='step_counter')
            is_training_and_counter_positive = tf.where(tf.greater(step_counter,0), self.is_training,tf.constant(True))

        mlp_builder = lambda input, widths,name=None: MultiLayerPerceptron(input, widths,with_batch_norm=self.hnet_hparams.with_batchnorm,scale=np.square(self.hnet_hparams.initialization_std),batchnorm_decay=self.hnet_hparams.batchnorm_decay,is_training=is_training_and_counter_positive,name=name,zero_fixer=self.hnet_hparams.zero_fixer)[0]

        with tf.device(next(self.gpus)):
            e_layer_outputs = mlp_builder(self.z, self.hnet_hparams.e_layer_sizes,'extractor')
            codes = {}
            for i,l in enumerate(layers):
                code_size = self.hnet_hparams.code_size_formula(num_of_weights_per_filter[l],num_of_filters[l])
                codes[l] = mlp_builder(e_layer_outputs[-1],[num_of_filters[l]*code_size],'codes{:d}'.format(i))
                codes[l] = tf.reshape(codes[l],[-1,num_of_filters[l],code_size],'codes{:d}'.format(i))

        wg_layer_outputs={}
        for i,l in enumerate(layers):
            with tf.device(next(self.gpus)):
                layer_widths = [self.hnet_hparams.wg_hidden_layer_size_formula(num_of_weights_per_filter[l],num_of_filters[l])]*self.hnet_hparams.wg_number_of_hidden_layers
                wg_layer_outputs_ = mlp_builder(codes[l],layer_widths + [num_of_weights_per_filter[l]],'weight_generator{:d}'.format(i))
                wg_layer_outputs[l] = wg_layer_outputs_
                layer_output = tf.transpose(wg_layer_outputs[l][-1],[0,2,1])
                if l.b_shape is None:
                    l.w = tf.reshape(layer_output,[-1]+l.w_shape)
                else:
                    l.w = tf.reshape(layer_output[:,:-1,:], [-1]+l.w_shape)
                    l.b = layer_output[:,-1,:]
                # TODO: change this if want to generate also BN!!
                if l.bn_scale_shape is not None:
                    l.bn_scale = tf.ones([batch_size] + l.bn_scale_shape)
                if l.bn_offset_shape is not None:
                    l.bn_offset = tf.zeros([batch_size] + l.bn_offset_shape)

        with tf.device(self.cpu):
            # TODO: change this if want to generate also BN!!
            flattened_network = weights.AsList(with_batchnorm=False)
            flattened_network = [tf.reshape(a,[batch_size]+[-1]) for a in flattened_network if a is not None]
            flattened_network = tf.concat(flattened_network,1)

        self.weights = weights
        self.layers = layers
        self.step_counter = step_counter
        self.batch_size = batch_size
        self.e_layer_outputs = e_layer_outputs
        self.codes = codes
        self.wg_layer_outputs = wg_layer_outputs
        self.flattened_network = flattened_network

        return weights

    def __AddOptimizationOps(self):

        with tf.device(next(self.gpus)):
            target = Resnet(self.x,self.target_hparams,self.image_params,self.y,weights=self.weights,order='NHWC',batch_type='BATCH_TYPE1')

            accuracy_loss = tf.reduce_mean(target.loss)
            accuracy = target.average_accuracy

        with tf.device(next(self.gpus)):
            mutual_distances = tf.reduce_sum(tf.abs(tf.expand_dims(self.flattened_network, 0) - tf.expand_dims(self.flattened_network, 1)), 2,name='mutual_squared_distances')
            nearest_distances = tf.identity(-1 * tf.nn.top_k(-1 * mutual_distances, k=2)[0][:, 1],name='nearest_distances')
            entropy_estimate = tf.identity(self.hnet_hparams.input_noise_size * tf.reduce_mean(tf.log(nearest_distances + self.hnet_hparams.zero_fixer)) + tf.digamma(tf.cast(self.batch_size, tf.float32)), name='entropy_estimate')

            diversity_loss = tf.identity(- 1 * entropy_estimate, name='diversity_loss')

            loss = tf.identity(self.hnet_hparams.lamBda* accuracy_loss + diversity_loss, name='loss')

        with tf.device(self.cpu):
            learning_rate = tf.Variable(self.hnet_hparams.learning_rate, dtype=tf.float32, trainable=False, name='learning_rate')
            learning_rate_rate = tf.Variable(self.hnet_hparams.learning_rate_rate, dtype=tf.float32, trainable=False,name='learning_rate_rate')
            update_learning_rate = tf.assign(learning_rate, learning_rate * learning_rate_rate,name='update_learning_rate')
            steps_before_train_step = [update_learning_rate]

        optimizer = tf.train.MomentumOptimizer(learning_rate,self.hnet_hparams.momentum)
        with tf.control_dependencies(steps_before_train_step):
            train_step = optimizer.minimize(loss, name='train_step',colocate_gradients_with_ops=True)
            with tf.control_dependencies([train_step]):
                step_counter_update = tf.assign_add(self.step_counter, 1, name='step_counter_update')
        train_step = tf.group(*(steps_before_train_step + [train_step, step_counter_update]), name='update_and_train')
        reset_optimizer = OptimizerReset(optimizer,self.graph,'resetter')

        initializer = tf.variables_initializer(self.graph.get_collection('variables'), name='initializer')

        saver = tf.train.Saver(max_to_keep=100)

        self.target = target
        self.accuracy_loss = accuracy_loss
        self.accuracy = accuracy
        self.mutual_distances = mutual_distances
        self.nearest_distances = nearest_distances
        self.entropy_estimate = entropy_estimate
        self.diversity_loss = diversity_loss
        self.loss = loss
        self.learning_rate = learning_rate
        self.learning_rate_rate = learning_rate_rate
        self.update_learning_rate = update_learning_rate
        self.optimizer = optimizer
        self.step_counter_update = step_counter_update
        self.train_step = train_step
        self.reset_optimizer = reset_optimizer
        self.initializer = initializer
        self.saver = saver

    def TrainStep(self,sess:tf.Session,additional_tensors_to_run=[]):
        z = self.SampleInput(self.hnet_hparams.batch_size)
        tensors_to_run = [self.train_step,self.step_counter]+additional_tensors_to_run
        out = sess.run(tensors_to_run,feed_dict={self.z:z})
        out.pop(0)
        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)

    def Train(self,sess:tf.Session,max_steps,initialize_from_checkpoint=False,checkpoint_file_name:str=None,restore_message:str=None):
        logger = GetLogger(initialize_from_checkpoint,checkpoint_file_name)
        if initialize_from_checkpoint:
            i = self.Restore(sess,checkpoint_file_name,logger,restore_message)
        else:
            i = self.Initialize(sess)
        while i<=max_steps:
            if i%100 == 0:
                i, accuracy, accuracy_loss, diversity_loss, total_loss = self.TrainStep(sess,[self.accuracy,self.accuracy_loss,self.diversity_loss,self.loss])
                logger.info("step {:d}: accuracy >>>{:.4f}<<<".format(i, accuracy))
                logger.info('  (accuracy_loss, diversity_loss, total_loss): ({:.5f}, {:.5f} ,{:.5f})'.format(accuracy_loss, diversity_loss,total_loss))
                self.UpdateStuff(sess,update_dict,logger)
            else:
                i = self.TrainStep(sess)

    def GetStepCounter(self,sess:tf.Session):
        return sess.run(self.step_counter)

    def Initialize(self,sess:tf.Session):
        sess.run(self.initializer)

        return self.GetStepCounter(sess)

    def Restore(self,sess:tf.Session, file_name,logger:Logger=None,restore_message:str=None):
        variables = self.graph.get_collection('variables')
        reader = tf.train.NewCheckpointReader(file_name)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], variables), variables))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        self.Initialize(sess)
        if os.path.isdir(file_name):
            saver.restore(sess, tf.train.latest_checkpoint(file_name))
        else:
            saver.restore(sess, file_name)

        step = self.GetStepCounter(sess)

        if logger is not None:
            logger.info("\n")
            logger.info("======INITIALIZED FROM CHECKPOINT======")
            logger.info("RETURNED TO STEP {:d}".format(step))
            if restore_message is not None:
                logger.info("additional message:  " + restore_message + "\n")
                logger.info("\n")

        return step

    def UpdateVariableFromFile(self, sess:tf.Session, var, filename):
        try:
            with open(filename) as fl:
                lines = fl.readlines()
            new_value = float(lines[0].strip())
            current_value = sess.run(var)
            sess.run(self.reset_optimizer)
            sess.run(var.assign(new_value))
            os.remove(filename)
            return new_value,current_value
        except FileNotFoundError:
            return None,None

    def UpdateLearningRateFromFile(self, sess:tf.Session, filename,logger:Logger=None):
        new, old = self.UpdateVariableFromFile(sess, self.learning_rate, filename)
        if logger is not None and new is not None:
            logger.info("--> Changed learning rate from {:.3e} to {:.3e} (factor of {:.1f})".format(old,new,max([old/new, new/old])))
        return new, old

    def UpdateLearningRateRateFromFile(self, sess:tf.Session, filename,logger:Logger=None):
        new, old = self.UpdateVariableFromFile(sess, self.learning_rate_rate, filename)
        if logger is not None and new is not None:
            logger.info("--> Changed learning rate rate from {:.6e} to {:.6e} (factor of {:.5f})".format(old, new, max([old/new, new/old])))

        return new, old

    def UpdateStuff(self,sess:tf.Session, filename_dict:dict,logger:Logger=None):
        self.UpdateLearningRateFromFile(sess,filename_dict['learning_rate'],logger)
        self.UpdateLearningRateRateFromFile(sess, filename_dict['learning_rate_rate'], logger)

    def SampleInput(self,batch_size=None):
        return np.random.uniform(-1*self.hnet_hparams.input_noise_bound,self.hnet_hparams.input_noise_bound,[batch_size,self.hnet_hparams.input_noise_size])

    def GenerateWeights(self,sess:tf.Session,z=None,batch_size=None,flatten=False):
        if z is None:
            if batch_size is None:
                batch_size = 1
            z = self.SampleInput(batch_size)
        if flatten:
            return sess.run(self.flattened_network, feed_dict={self.z: z})
        else:
            return sess.run(self.weights, feed_dict={self.z: z})

    def SaveToCheckpoint(self,sess:tf.Session,filename):
        self.saver.save(sess,filename,global_step=self.GetStepCounter(sess))