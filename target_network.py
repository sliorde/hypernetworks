import tensorflow as tf
from params import ResNetHyperParameters, DataParams
from utils import ConvBN,MaxPool
from logging import Logger
from resnet_weights import ResnetWeights

class Resnet():
    def __init__(self, input, hparams:ResNetHyperParameters, image_params:DataParams, labels=None, train=False, weights:ResnetWeights=None, noise_batch_size=1, graph=None):
        """
        a resnet model that can be trained and used for inference. The weight for the model can either be created as variables from within this object, or given from outside (for example, from a hypernetwork generator)
        Args:
            input: the input images tensor
            hparams:
            image_params:
            labels: the one-hot labels. this is optional, and needed only for training or evaluation purposes
            train: whether to add training ops
            weights: optional - a `ResnetWeights` object with all the weights for the resnet. if `None`, the new weights will be initialized
            noise_batch_size:
            batch_type: one of `'BATCH_TYPE1'`,`'BATCH_TYPE2'`,`'BATCH_TYPE3'`,`'BATCH_TYPE4'`. In the first option, the batch dimension of the weights and the images will be shared. In the second option, the imagse will be two batch dimensions: one for noise and one for images, and the noise batch dimension will be shared with the weights. In the third option, the images and the weights will both have a non-shared batch dimension. In the fourth option - images will have a batch dimension, but weight will not.
            noise_batch_size: this will be ignored, unless `weights` is `None` and `batch_type` is `'BATCH_TYPE3'`, in which case it will determine the batch size for the weights.
            graph:
        """
        self.order = image_params.order
        self.batch_type = hparams.batch_type
        self.labels = labels
        self.input = input

        self.hparams = hparams

        self.image_params = image_params

        if self.order not in {'NCHW','NHWC'}:
            raise ValueError("invalid value for `order`")

        if self.batch_type.upper() not in {'BATCH_TYPE1','BATCH_TYPE2','BATCH_TYPE3','BATCH_TYPE4','BATCH_TYPE5'}: # TODO clean up
            raise ValueError("invalid value for `batch_type`")

        if graph is None:
            graph = tf.get_default_graph()
        with graph.as_default():
            if weights is None: # if weights were not supplied, create new variables
                if self.batch_type in ['BATCH_TYPE1','BATCH_TYPE2','BATCH_TYPE5']: # TODO clean up
                    noise_batch_size = input.shape[0]
                elif self.batch_type=='BATCH_TYPE4':
                    noise_batch_size = None
                weights = self.__CreateWeightVariables(noise_batch_size)
            self.weights = weights

            noise_batch_size = tf.shape(weights['first_layer']['w'])[0]
            if self.batch_type=='BATCH_TYPE3': # this in effect converts BATCH_TYPE3 to BATCH_TYPE2
                input = tf.tile(tf.expand_dims(input,0), [noise_batch_size,1,1,1,1])

            # build network
            self.__Build(input)
            if labels is not None:
                self.__AddLossOp()
            if train:
                self.__AddTrainingOps()
        self.graph = graph

    def __Build(self,input):

        weights = self.weights

        num_scales = len(self.hparams.num_blocks_per_scale)

        x = tf.nn.relu(ConvBN(input, weights['first_layer']['w'], self.hparams.stride_between_scales, weights['first_layer']['bn_scale'], weights['first_layer']['bn_offset'],self.order,self.batch_type))

        if self.batch_type in ['BATCH_TYPE2','BATCH_TYPE3']:
            x = tf.map_fn(lambda xx: MaxPool(xx, self.hparams.first_layer_pool_size,self.hparams.stride_between_scales,self.order),x)
        else:
            x = MaxPool(x, self.hparams.first_layer_pool_size,self.hparams.stride_between_scales,self.order)

        for i in range(num_scales):
            for j in range(self.hparams.num_blocks_per_scale[i]):
                if (i > 0) and (j == 0): # condition checks whether to apply a projection in residual connection
                    x1 = ConvBN(x, weights['scales'][i]['blocks'][0]['projection']['w'], self.hparams.stride_between_scales, weights['scales'][i]['blocks'][0]['projection']['bn_scale'], weights['scales'][i]['blocks'][0]['projection']['bn_offset'],self.order,self.batch_type)
                else:
                    x1 = x
                for k in range(self.hparams.layers_per_block):
                    if (j == 0) and (k == 0) and (i > 0): # condition checks whether to apply a stride in the convolution
                        stride = self.hparams.stride_between_scales
                    else:
                        stride = 1
                    x = ConvBN(x, weights['scales'][i]['blocks'][j]['conv_layers'][k]['w'], stride, weights['scales'][i]['blocks'][j]['conv_layers'][k]['bn_scale'], weights['scales'][i]['blocks'][j]['conv_layers'][k]['bn_offset'],self.order,self.batch_type)
                    if (k < (self.hparams.layers_per_block - 1)):
                        x = tf.nn.relu(x)

                x = tf.nn.relu(x + x1)

        if self.order=='NHWC':
            x = tf.reduce_mean(x,[-3,-2])
        else: # 'NCHW'
            x = tf.reduce_mean(x, [-2, -1])

        x = tf.expand_dims(x, -1)
        if self.batch_type=='BATCH_TYPE1' or self.batch_type=='BATCH_TYPE5': # TODO clean up
            w = weights['final_layer']['w']
        elif self.batch_type in ['BATCH_TYPE2','BATCH_TYPE3']:
            w = tf.expand_dims(weights['final_layer']['w'],1)
        else:
            w = tf.expand_dims(weights['final_layer']['w'], 0)
        if self.batch_type in ['BATCH_TYPE2', 'BATCH_TYPE3']:
            b = tf.expand_dims(weights['final_layer']['b'], 1)
        else:
            b = weights['final_layer']['b']

        x = tf.reduce_sum(x * w,-2) + b
        self.logits = x
        self.probabilities = tf.nn.softmax(self.logits, -1)
        self.predictions = tf.argmax(self.probabilities, axis=-1)

    def __AddLossOp(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=self.labels),-1)
        self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, axis=-1))
        self.average_accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

    def __AddTrainingOps(self):
        self.step_counter = tf.Variable(0, trainable=False)
        l2_loss = 0 # maybe we will add regularization in the future
        self.loss_with_regularization = tf.reduce_mean(self.loss + l2_loss)
        self.train_step = tf.train.MomentumOptimizer(learning_rate=self.hparams.learning_rate, momentum=self.hparams.momentum).minimize(self.loss_with_regularization)
        with tf.control_dependencies([self.train_step]):
            self.step_counter_update = tf.assign_add(self.step_counter, 1)
        self.train_step = tf.group(self.train_step, self.step_counter_update)

    def RunTrainStep(self,sess:tf.Session=None):
        if sess is None:
            sess = tf.get_default_session()
        loss,_ = sess.run([self.loss,self.train_step])
        return loss

    def Train(self,sess:tf.Session=None,logger:Logger=None):
        for i in range(50000):
            loss = self.RunTrainStep(sess)
            if (i%10 == 0):
                if logger is not None:
                    logger.info('step {:d},  training loss:{:.4f}'.format(i,loss))

    def Predict(self,input,weights=None,sess:tf.Session=None):
        if sess is None:
            sess = tf.get_default_session()
        feed_dict = {}
        if weights is not None:
            feed_dict = self.weights.CreateFeedDict(weights)
        feed_dict[self.input] = input
        return sess.run(self.predictions,feed_dict=feed_dict)

    def GetLoss(self,input,labels,weights=None,sess:tf.Session=None):
        if sess is None:
            sess = tf.get_default_session()
        feed_dict = {}
        if weights is not None:
            feed_dict = self.weights.CreateFeedDict(weights)
        feed_dict[self.input] = input
        feed_dict[self.labels] = labels
        return sess.run(self.loss,feed_dict=feed_dict)

    def __CreateWeightVariables(self,batch_size=None):
        """
        create a `ResnetWeights` object, populated with `tf.Variable`s.
        Args:
            batch_size:

        Returns:

        """
        if batch_size is None:
            batch_dim = []
        else:
            batch_dim = [batch_size]
        if self.order == 'NCHW':
            permute = lambda u: [u[2],u[0],u[1]]
        else:
            permute = lambda u:u
        weights = ResnetWeights(self.hparams,self.image_params)
        num_scales = len(self.hparams.num_blocks_per_scale)
        initializer = tf.variance_scaling_initializer()

        # first layer
        weights['first_layer']['w']=tf.get_variable('initial_layer_w',batch_dim+weights['first_layer'].w_shape , tf.float32, initializer=initializer)
        weights['first_layer']['bn_scale']=tf.get_variable('initial_layer_bn_scale', batch_dim+permute(weights['first_layer'].bn_scale_shape), tf.float32, initializer=tf.ones_initializer())
        weights['first_layer']['bn_offset']=tf.get_variable('initial_layer_bn_offset',batch_dim+permute(weights['first_layer'].bn_offset_shape), tf.float32, initializer=tf.zeros_initializer())

        # scale layers
        for i in range(num_scales):
            for j in range(self.hparams.num_blocks_per_scale[i]):

                # add projections layers to residual connections
                if (i > 0) and (j == 0):
                    weights['scales'][i]['blocks'][0]['projection']['w']=tf.get_variable('scale{:d}_block{:d}_projection_w'.format(i,j),batch_dim+weights['scales'][i]['blocks'][0]['projection'].w_shape,tf.float32,initializer=initializer)
                    weights['scales'][i]['blocks'][0]['projection']['bn_scale'] = tf.get_variable('scale{:d}_block{:d}_projection_bn_scale'.format(i,j),batch_dim+permute(weights['scales'][i]['blocks'][0]['projection'].bn_scale_shape),tf.float32,initializer=tf.ones_initializer())
                    weights['scales'][i]['blocks'][0]['projection']['bn_offset'] = tf.get_variable('scale{:d}_block{:d}_projection_bn_offset'.format(i,j),batch_dim+permute(weights['scales'][i]['blocks'][0]['projection'].bn_offset_shape),tf.float32,initializer=tf.zeros_initializer())

                # add convolutional layers to block
                for k in range(self.hparams.layers_per_block):
                    weights['scales'][i]['blocks'][j]['conv_layers'][k]['w'] = tf.get_variable('scale{:d}_block{:d}_layer{:d}_w'.format(i,j,k),batch_dim+weights['scales'][i]['blocks'][j]['conv_layers'][k].w_shape,tf.float32,initializer=initializer)
                    weights['scales'][i]['blocks'][j]['conv_layers'][k]['bn_scale'] = tf.get_variable('scale{:d}_block{:d}_layer{:d}_bn_scale'.format(i,j,k),batch_dim+permute(weights['scales'][i]['blocks'][j]['conv_layers'][k].bn_scale_shape),tf.float32,initializer=tf.ones_initializer())
                    weights['scales'][i]['blocks'][j]['conv_layers'][k]['bn_offset'] = tf.get_variable('scale{:d}_block{:d}_layer{:d}_bn_offset'.format(i,j,k),batch_dim+permute(weights['scales'][i]['blocks'][j]['conv_layers'][k].bn_offset_shape),tf.float32,initializer=tf.zeros_initializer())

        # final layer
        weights['final_layer']['w'] = tf.get_variable('final_layer_w',batch_dim+weights['final_layer'].w_shape,tf.float32,initializer=initializer)
        weights['final_layer']['b'] = tf.get_variable('final_layer_b',batch_dim+weights['final_layer'].b_shape,tf.float32,initializer=tf.zeros_initializer())

        return weights

    def NumberOfWeights(self,with_batchnorm:bool=True):
        return self.weights.NumberOfWeights(with_batchnorm)