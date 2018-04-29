import tensorflow as tf
import numpy as np
from params import ResNetHyperParameters, DataParams
from utils import ConvBN,MaxPool
from logging import Logger

class ResnetWeights():
    def __init__(self, hparams:ResNetHyperParameters, image_params:DataParams):
        class WeightedLayer():
            def __init__(self,w_shape=None,b_shape=None,bn_scale_shape=None,bn_offset_shape=None):
                self.w = None
                self.w_shape = w_shape
                self.b = None
                self.b_shape = b_shape
                self.bn_scale = None
                self.bn_scale_shape = bn_scale_shape
                self.bn_offset = None
                self.bn_offset_shape = bn_offset_shape
            def AsList(self,shapes:bool=False,with_filters:bool=True,with_batchnorm:bool=True):
                l = []
                if shapes:
                    if with_filters:
                        l.extend([self.w_shape, self.b_shape])
                    if with_batchnorm:
                        l.extend([self.bn_scale_shape, self.bn_offset_shape])
                else:
                    if with_filters:
                        l.extend([self.w, self.b])
                    if with_batchnorm:
                        l.extend([self.bn_scale, self.bn_offset])
                return l

        class ResnetScale():
            def __init__(self, scale_ind:int, hparams:ResNetHyperParameters, image_params:DataParams):
                class ResnetBlock():
                    def __init__(self, block_ind:int, scale_ind:int, hparams:ResNetHyperParameters, image_params:DataParams):
                        width = hparams.layer1_size*(hparams.stride_between_scales ** scale_ind)
                        if scale_ind>0 and block_ind==0:
                            prev_width = hparams.layer1_size*(hparams.stride_between_scales ** (scale_ind-1))
                        else:
                            prev_width = width
                        size = image_params.image_size//(hparams.stride_between_scales**(scale_ind+2))

                        if (scale_ind > 0) and (block_ind == 0):

                            w_shape = [1,1,int(prev_width),int(width)]
                            bn_scale_shape = [int(size),int(size),int(width)]
                            bn_offset_shape = bn_scale_shape
                            self.projection = WeightedLayer(w_shape=w_shape,bn_scale_shape=bn_scale_shape,bn_offset_shape=bn_offset_shape)
                        else:
                            self.projection = WeightedLayer(None)

                        self.conv_layers = []
                        for k in range(hparams.layers_per_block):
                            if k==0:
                                w_shape = [int(hparams.filter_size),int(hparams.filter_size),int(prev_width),int(width)]
                            else:
                                w_shape = [int(hparams.filter_size), int(hparams.filter_size), int(width), int(width)]
                            bn_scale_shape = [int(size),int(size),int(width)]
                            bn_offset_shape = bn_scale_shape
                            self.conv_layers.append(WeightedLayer(w_shape=w_shape,bn_scale_shape=bn_scale_shape,bn_offset_shape=bn_offset_shape))

                    def __getitem__(self, item):
                        return self.conv_layers[item]

                    def AsList(self,shapes:bool=False,with_filters:bool=True,with_batchnorm:bool=True):
                        weights = []
                        weights.extend(self.projection.AsList(shapes,with_filters,with_batchnorm))
                        for l in self.conv_layers:
                            weights.extend(l.AsList(shapes,with_filters,with_batchnorm))
                        return weights

                self.blocks = []
                for block_ind in range(hparams.num_blocks_per_scale[scale_ind]):
                    self.blocks.append(ResnetBlock(block_ind,scale_ind,hparams,image_params))

            def __getitem__(self, item):
                return self.blocks[item]

            def AsList(self,shapes:bool=False,with_filters:bool=True,with_batchnorm:bool=True):
                weights = []
                for a in self.blocks:
                    weights.extend(a.AsList(shapes,with_filters,with_batchnorm))
                return weights

        size = image_params.image_size/hparams.stride_between_scales
        w_shape = [int(hparams.layer1_filter_size),int(hparams.layer1_filter_size),int(image_params.number_of_channels),int(hparams.layer1_size)]
        bn_scale_shape = [int(size),int(size),int(hparams.layer1_size)]
        bn_offset_shape = bn_scale_shape
        self.layer1 = WeightedLayer(w_shape=w_shape, bn_scale_shape=bn_scale_shape, bn_offset_shape=bn_offset_shape)

        self.scales = []
        for scale_ind in range(len(hparams.num_blocks_per_scale)):
            self.scales.append(ResnetScale(scale_ind,hparams,image_params))

        w_shape = [int(hparams.layer1_size*(hparams.stride_between_scales**(len(hparams.num_blocks_per_scale)-1))),int(image_params.num_classes)]
        b_shape = [int(image_params.num_classes)]
        self.final_layer = WeightedLayer(w_shape=w_shape,b_shape=b_shape)

    def AsList(self,shapes:bool=False,with_filters:bool=True,with_batchnorm:bool=True):
        weights = []
        weights.extend(self.layer1.AsList(shapes,with_filters,with_batchnorm))
        for a in self.scales:
            weights.extend(a.AsList(shapes,with_filters,with_batchnorm))
        weights.extend(self.final_layer.AsList(shapes,with_filters,with_batchnorm))
        return weights

    def NumberOfWeights(self):
        shapes = self.AsList(shapes=True)
        return np.sum([np.prod(u) for u in shapes if u is not None])

    def WeightedLayerIterator(self):
        scale = 0
        while scale <= len(self.scales)+1:
            if scale==0:
                yield self.layer1
            elif scale==(len(self.scales)+1):
                yield self.final_layer
            else:
                block = 0
                while block < len(self.scales[scale-1].blocks):
                    layer = 0
                    while layer<len(self.scales[scale-1].blocks[block].conv_layers):
                        yield self.scales[scale-1].blocks[block].conv_layers[layer]
                        layer += 1
                    if self.scales[scale-1].blocks[block].projection.w_shape is not None:
                        yield self.scales[scale-1].blocks[block].projection
                    block += 1
            scale += 1


class Resnet():
    def __init__(self, input, hparams:ResNetHyperParameters, image_params:DataParams, labels=None, train=False, weights:ResnetWeights=None, noise_batch_size=1, order='NCHW', batch_type='BATCH_TYPE1', graph=None):
        self.order = order
        self.batch_type = batch_type
        self.labels = labels
        self.input = input

        self.hparams = hparams

        self.image_params = image_params

        if order not in {'NCHW','NHWC'}:
            raise ValueError("invalid value for `order`")

        if batch_type not in {'BATCH_TYPE1','BATCH_TYPE2','BATCH_TYPE3','BATCH_TYPE4'}:
            raise ValueError("invalid value for `batch_type`")

        if graph is None:
            graph = tf.get_default_graph()
        with graph.as_default():
            if weights is None:
                if batch_type in ['BATCH_TYPE1','BATCH_TYPE2']:
                    noise_batch_size = input.shape[0]
                elif batch_type=='BATCH_TYPE4':
                    noise_batch_size = None
                weights = self.__CreateWeightVariables(noise_batch_size)
            self.weights = weights
            noise_batch_size = tf.shape(weights.layer1.w)[0]
            if self.batch_type=='BATCH_TYPE3':
                input = tf.tile(tf.expand_dims(input,0), [noise_batch_size,1,1,1,1])
            self.__Build(input)
            if labels is not None:
                self.__AddLossOp()
            if train:
                self.__AddTrainingOps()
        self.graph = graph

    def __Build(self,input):

        weights = self.weights

        num_scales = len(self.hparams.num_blocks_per_scale)

        x = tf.nn.relu(ConvBN(input, weights.layer1.w, self.hparams.stride_between_scales, weights.layer1.bn_scale, weights.layer1.bn_offset,self.order,self.batch_type))

        if self.batch_type in ['BATCH_TYPE2','BATCH_TYPE3']:
            x = tf.map_fn(lambda xx: MaxPool(xx, self.hparams.layer1_pool_size,self.hparams.stride_between_scales,self.order),x)
        else:
            x = MaxPool(x, self.hparams.layer1_pool_size,self.hparams.stride_between_scales,self.order)

        for i in range(num_scales):
            for j in range(self.hparams.num_blocks_per_scale[i]):
                if (i > 0) and (j == 0):
                    x1 = ConvBN(x, weights.scales[i].blocks[0].projection.w, self.hparams.stride_between_scales, weights.scales[i].blocks[0].projection.bn_scale, weights.scales[i].blocks[0].projection.bn_offset,self.order,self.batch_type)
                else:
                    x1 = x
                for k in range(self.hparams.layers_per_block):
                    if (j == 0) and (k == 0) and (i > 0):
                        stride = self.hparams.stride_between_scales
                    else:
                        stride = 1
                    x = ConvBN(x, weights.scales[i].blocks[j].conv_layers[k].w, stride, weights.scales[i].blocks[j].conv_layers[k].bn_scale, weights.scales[i].blocks[j].conv_layers[k].bn_offset,self.order,self.batch_type)
                    if (k < (self.hparams.layers_per_block - 1)):
                        x = tf.nn.relu(x)

                x = tf.nn.relu(x + x1)

        if self.order=='NHWC':
            x = tf.reduce_mean(x,[-3,-2])
        else: # 'NCHW'
            x = tf.reduce_mean(x, [-2, -1])

        x = tf.expand_dims(x, -1)
        if self.batch_type=='BATCH_TYPE1':
            w = weights.final_layer.w
        elif self.batch_type in ['BATCH_TYPE2','BATCH_TYPE3']:
            w = tf.expand_dims(weights.final_layer.w,1)
        else:
            w = tf.expand_dims(weights.final_layer.w, 0)
        if self.batch_type in ['BATCH_TYPE2', 'BATCH_TYPE3']:
            b = tf.expand_dims(weights.final_layer.b, 1)
        else:
            b = weights.final_layer.b

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
        l2_loss = 0
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
            feed_dict = self.__CreateWeightsFeedDict(weights)
        feed_dict[self.input] = input
        return sess.run(self.predictions,feed_dict=feed_dict)

    def GetLoss(self,input,labels,weights=None,sess:tf.Session=None):
        if sess is None:
            sess = tf.get_default_session()
        feed_dict = {}
        if weights is not None:
            feed_dict = self.__CreateWeightsFeedDict(weights)
        feed_dict[self.input] = input
        feed_dict[self.labels] = labels
        return sess.run(self.loss,feed_dict=feed_dict)

    def __CreateWeightsFeedDict(self,weights:ResnetWeights):
        feed_dict = {}
        self_weights = self.weights.AsList()
        weights = weights.AsList()
        for _,i in enumerate(self_weights):
            if (self_weights[i] is not None) and (weights[i] is not None):
                feed_dict[self_weights[i]] = weights[i]
        return feed_dict

    def __CreateWeightVariables(self,batch_size=None):
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
        weights.layer1.w = tf.get_variable('initial_layer_w',batch_dim+weights.layer1.w_shape , tf.float32, initializer=initializer)
        weights.layer1.bn_scale = tf.get_variable('initial_layer_bn_scale', batch_dim+permute(weights.layer1.bn_scale_shape), tf.float32, initializer=tf.ones_initializer())
        weights.layer1.bn_offset = tf.get_variable('initial_layer_bn_offset',batch_dim+permute(weights.layer1.bn_offset_shape), tf.float32, initializer=tf.zeros_initializer())
        for i in range(num_scales):
            for j in range(self.hparams.num_blocks_per_scale[i]):
                if (i > 0) and (j == 0):
                    weights.scales[i].blocks[0].projection.w = tf.get_variable('scale{:d}_block{:d}_projection_w'.format(i,j),batch_dim+weights.scales[i].blocks[0].projection.w_shape,tf.float32,initializer=initializer)
                    weights.scales[i].blocks[0].projection.bn_scale = tf.get_variable('scale{:d}_block{:d}_projection_bn_scale'.format(i,j),batch_dim+permute(weights.scales[i].blocks[0].projection.bn_scale_shape),tf.float32,initializer=tf.ones_initializer())
                    weights.scales[i].blocks[0].projection.bn_offset = tf.get_variable('scale{:d}_block{:d}_projection_bn_offset'.format(i,j),batch_dim+permute(weights.scales[i].blocks[0].projection.bn_offset_shape),tf.float32,initializer=tf.zeros_initializer())
                for k in range(self.hparams.layers_per_block):
                    weights.scales[i].blocks[j].conv_layers[k].w = tf.get_variable('scale{:d}_block{:d}_layer{:d}_w'.format(i,j,k),batch_dim+weights.scales[i].blocks[j].conv_layers[k].w_shape,tf.float32,initializer=initializer)
                    weights.scales[i].blocks[j].conv_layers[k].bn_scale = tf.get_variable('scale{:d}_block{:d}_layer{:d}_bn_scale'.format(i,j,k),batch_dim+permute(weights.scales[i].blocks[j].conv_layers[k].bn_scale_shape),tf.float32,initializer=tf.ones_initializer())
                    weights.scales[i].blocks[j].conv_layers[k].bn_offset = tf.get_variable('scale{:d}_block{:d}_layer{:d}_bn_offset'.format(i,j,k),batch_dim+permute(weights.scales[i].blocks[j].conv_layers[k].bn_offset_shape),tf.float32,initializer=tf.zeros_initializer())
        weights.final_layer.w = tf.get_variable('final_layer_w',batch_dim+weights.final_layer.w_shape,tf.float32,initializer=initializer)
        weights.final_layer.b = tf.get_variable('final_layer_b',batch_dim+weights.final_layer.b_shape,tf.float32,initializer=tf.zeros_initializer())

        return weights