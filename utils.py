import os
import sys
import types
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import logging
from datetime import datetime

def GetWeightVariable(shape, name, scale=1.0):
    return tf.get_variable(shape=shape, initializer=tf.variance_scaling_initializer(scale=scale),name=name)

def GetBiasVariable(shape, name, scale=1.0):
    return tf.get_variable(shape=shape, initializer=tf.constant_initializer(0.0),trainable=False,name=name)

def MultiLayerPerceptron(input,widths,with_batch_norm=False,where_to_batch_norm=None,train_batch_norm=True,activation_on_last_layer=False,is_training=None,scale=1.0,batchnorm_decay=0.98,activation=tf.nn.relu,name=None,zero_fixer=1e-8):
    """
    Creates a multilayer perceptron (=MLP)
    Args:
        input: input layer to perceptron
        widths: a list with the widths (=number of neurons) in each layer
        with_batch_norm: should perfrom batch normalization?
        where_to_batch_norm: (optional) a list with boolean values, one value for each layer. This determines whether to perform batch normalization in this layer.
        train_batch_norm: (optional) whether to train the batch normalization's offsets and scales, or just use constant values
        activation_on_last_layer: if boolean - whether to add activation on the last layer of the MLP. Otherwise, can be an actual activation function for the last layer
        is_training: `bool` or boolean `tf.Variable` which determines whether currently the graph is in training phase. This determines which batch normalization value to use.
        scale: (optional) the variance of the initial values of the variables will be proportional to this
        batchnorm_decay: exponential decay constant for batch normalization
        activation:
        name:
        zero_fixer: small numnber to be used for division by zero etc.

    Returns:
        three lists:
            layer_outputs: activation maps tensors
            layers: tuples (w,b) of variable tensors
            batch_norm_params: tuples (means,variances,offsets,scales) for batch norm parameters
    """
    inds = np.nonzero(widths)[0]
    widths = [widths[ind] for ind in inds]
    layers = []
    layer_outputs = [input,]
    widths = [int(input.shape[-1])]+widths
    if with_batch_norm and is_training is not None:
        if where_to_batch_norm is None:
            where_to_batch_norm = [False] + [True]*(len(widths)-1)
            if activation_on_last_layer is False:
                where_to_batch_norm[-1] = False
        else:
            where_to_batch_norm = [where_to_batch_norm[ind] for ind in inds]
        batch_norm_params = []
    else:
        batch_norm_params = None
        where_to_batch_norm = [False]*len(widths)
    for i in range(1,len(widths)):
        w = GetWeightVariable([widths[i - 1], widths[i]], GiveName(name,'layer{:d}_weights'.format(i)), scale)
        if where_to_batch_norm[i]:
            b = tf.zeros([widths[i]],name=GiveName(name,'layer{:d}_biases'.format(i)))
        else:
            b = GetBiasVariable([widths[i]], GiveName(name,'layer{:d}_biases'.format(i)), scale)
        layers.append((w, b))
        pre_activations = tf.add(tf.tensordot(layer_outputs[i - 1], w, [[-1], [-2]]), b, GiveName(name,'layer{:d}_pre_activations'.format(i)))
        if i<(len(widths)-1):
            layer_output = activation(pre_activations, GiveName(name,'layer{:d}_activations'.format(i)))
        else:
            if activation_on_last_layer is False:
                layer_output = pre_activations
            elif activation_on_last_layer is True:
                layer_output = activation(pre_activations,GiveName(name,'layer{:d}_activations'.format(i)))
            else:
                layer_output = activation_on_last_layer(pre_activations, GiveName(name,'layer{:d}_activations'.format(i)))
        if where_to_batch_norm[i]:
            layer_output,params = AddBacthNormalizationOps(layer_output,is_training,train_batch_norm,batchnorm_decay,zero_fixer,GiveName(name,'layer{:d}_batchnorm'.format(i)))
            batch_norm_params.append(params)
        layer_outputs.append(layer_output)
    layer_outputs = layer_outputs[1:]
    return layer_outputs,layers,batch_norm_params

def AddBacthNormalizationOps(input, is_training, train_BN_params, batchnorm_decay, zero_fixer=1e-8, name=None):
    """

    Args:
        input: input layer
        is_training: a bool or a tf boolean tensor which determines whether currently the graph is in training phase. This determines whether to use batch or aggregated mean and std
        train_BN_params: whether to train the batch normalization's offsets and scales, or just use constant values
        batchnorm_decay: exponential decay constant for batch normalization
        zero_fixer: small numnber to be used for division by zero etc.
        name:

    Returns:
        a tuple (output_layer, (means,variances,offsets,scales))
    """
    batch_means, batch_variances = tf.nn.moments(input, list(np.arange(0, len(input.shape) - 1)),keep_dims=False)
    offsets = tf.get_variable(name=GiveName(name,'offsets'),shape=batch_means.shape,initializer=tf.zeros_initializer(),trainable=train_BN_params)
    scales = tf.get_variable(name=GiveName(name,'scales'), shape=batch_variances.shape,initializer=tf.ones_initializer(),trainable=train_BN_params)
    ema_average_batch_means = tf.get_variable(name=GiveName(name,'ema'),shape=batch_means.shape,initializer=tf.zeros_initializer())
    ema_average_batch_variances = tf.get_variable(name=GiveName(name, 'ema'), shape=batch_variances.shape,initializer=tf.zeros_initializer())
    ema_batch_means_apply = tf.assign_sub(ema_average_batch_means,(ema_average_batch_means - batch_means) * batchnorm_decay)
    ema_batch_variances_apply = tf.assign_sub(ema_average_batch_variances,(ema_average_batch_variances - batch_variances) * batchnorm_decay)
    def ApplyEmaUpdate():
        with tf.control_dependencies([ema_batch_means_apply,ema_batch_variances_apply]):
            return tf.identity(batch_means), tf.identity(batch_variances)
    if isinstance(is_training,bool):
        if is_training:
            means, variances = ApplyEmaUpdate()
        else:
            means, variances = ema_average_batch_means, ema_average_batch_variances
    else: # in this case, is_training is a tf.Tensor of type tf.bool
        means, variances = tf.cond(is_training, lambda: ApplyEmaUpdate(),lambda: (ema_average_batch_means, ema_average_batch_variances))
    return tf.nn.batch_normalization(input, means, variances, offsets, scales,zero_fixer,name), (means,variances,offsets,scales)

def GiveName(name:str,more_info:str):
    if name is None:
        return None
    elif more_info is None:
        return name
    else:
        return name+'_'+more_info


def MaxPool(x,size,stride,order='NHWC'):
    if order=='NHWC':
        size = [1,size,size,1]
        stride = [1,stride,stride,1]
    else:
        size = [1, 1, size, size]
        stride = [1, 1, stride, stride]
    return tf.nn.max_pool(x,size,stride,'SAME',order)

def Conv2D(x,w,stride,order):
    if order=='NHWC':
        stride = [1,stride,stride,1]
    else:
        stride = [1, 1, stride, stride]
    return tf.nn.conv2d(x, w, stride, 'SAME', data_format=order)

def ConvType1(x, w, stride, order):
    # batch_size = tf.shape(w)[0]
    # w_sz = [batch_size] + w.shape.as_list()[1:]  # [batch,width,height,channels,filters]
    # w = tf.transpose(w, [1, 2, 0, 3, 4])  # [width,height,batch,channels,filters]
    # w = tf.reshape(w, [w_sz[1], w_sz[2], w_sz[0] * w_sz[3], w_sz[4]])  # [width,height,batch*channels,filters]
    # if order == 'NHWC':
    #     x_sz = [batch_size]+x.shape.as_list()[1:]  # [batch,width,height,channels]
    #     x = tf.transpose(x, [1, 2, 0, 3])  # [width,height,batch,channels]
    #     x = tf.reshape(x, [1, x_sz[1], x_sz[2], x_sz[0] * x_sz[3]])  # [1,width,height,batch*channels]
    #     x = tf.nn.depthwise_conv2d(x, w, strides=[1, stride, stride, 1],padding='SAME')  # [1,width/stride,height/stride,batch*channels*filters]
    #     x = tf.reshape(x, [x_sz[1]//stride, x_sz[2]//stride, x_sz[0], x_sz[3], w_sz[4]])  # [width/stride,height/stride,batch,channels,filters]
    #     x = tf.transpose(x, [2, 0, 1, 3, 4])  # [batch,width,height,channels,filters]
    #     x = tf.reduce_sum(x, axis=3)  # [batch,width,height,filters]
    # else:  # NHCW
    #     x_sz = [batch_size] + x.shape.as_list()[1:]  # [batch,channels,width,height]
    #     x = tf.reshape(x, [1, x_sz[0] * x_sz[1], x_sz[2], x_sz[3]])  # [1,batch*channels,width,height]
    #     x = tf.nn.depthwise_conv2d(x, w, strides=[1, 1, stride, stride],padding='SAME')  # [1,batch*channels*filters,width/stride,height/stride]
    #     x = tf.reshape(x, [x_sz[0], x_sz[1], w_sz[4], x_sz[2]//stride, x_sz[3]//stride])  # [batch,channels,filters,width/stride,height/stride]
    #     x = tf.reduce_sum(x, axis=1)  # [batch,filters,width,height]
    # return x
    return tf.squeeze(tf.map_fn(lambda u: Conv2D(tf.expand_dims(u[0],0), u[1], stride, order), elems=[x, w], dtype=tf.float32), 1)

def ConvType2(x, w, stride, order):
    return tf.map_fn(lambda u: Conv2D(u[0],u[1],stride,order), elems=[x, w],dtype=tf.float32)

def ConvType4(x, w, stride, order):
    return Conv2D(x, w, stride,order)

def ConvBN_old(x, w, stride, scale, offset, order, batch_type='BATCH_TYPE1', name=None):
    """
    perform convolution and "batch norm", where the batch norm does not calculate means and variances
    Args:
        x:
        w:
        stride:
        scale:
        offset:
        order:
        batch_type:
        name:

    Returns:

    """
    if batch_type in ['BATCH_TYPE2','BATCH_TYPE3']:
        scale = tf.expand_dims(scale, 1)
        offset = tf.expand_dims(offset, 1)

    conv_func = [ConvType1,ConvType2,ConvType2,ConvType4]
    batch_type = ['BATCH_TYPE1','BATCH_TYPE2','BATCH_TYPE3','BATCH_TYPE4'].index(batch_type)
    conv_func = conv_func[batch_type]

    return tf.identity(conv_func(x, w, stride, order) * scale + offset, name)

def ConvBN(x, w, stride, scale, offset, order, batch_type='BATCH_TYPE1', name=None):
    if batch_type in ['BATCH_TYPE2','BATCH_TYPE3']:
        scale = tf.expand_dims(scale, 1)
        offset = tf.expand_dims(offset, 1)

    conv_func = [ConvType1,ConvType2,ConvType2,ConvType4]
    batch_type_index = ['BATCH_TYPE1','BATCH_TYPE2','BATCH_TYPE3','BATCH_TYPE4','BATCH_TYPE5'].index(batch_type) # TODO clean up
    if batch_type_index == 4: batch_type_index = 0 # TODO clean up
    conv_func = conv_func[batch_type_index]

    x = conv_func(x, w, stride, order)

    # TODO clean up
    if batch_type == 'BATCH_TYPE5':
        # TODO tf.nn.moments() is slow, use tf.nn.fused_batch_norm() with cudnn
        axes = [0, 2, 3] if order == 'NCHW' else [0, 1, 2]
        means, variances = tf.map_fn(lambda u: tf.nn.moments(tf.expand_dims(u,0), axes=axes, keep_dims=True), elems=x, dtype=(tf.float32, tf.float32))
        x = tf.squeeze(tf.map_fn(lambda u: tf.nn.batch_normalization(x=tf.expand_dims(u[0],0), scale=u[1], offset=u[2], mean=u[3], variance=u[4], variance_epsilon=1e-5), elems=[x, scale, offset, means, variances], dtype=tf.float32), 1)

    return tf.identity(x, name)

def OptimizerReset(optimizer, graph=None, name=None):
    """
    reset all internal variables (=slots) of optimizer. It is important to do this  when doing a manual sharp change
    :param name:
    :return:
    """
    if graph is None:
        graph = tf.get_default_graph()
    slots = [optimizer.get_slot(var, name) for name in optimizer.get_slot_names() for var in graph.get_collection('variables')]
    slots = [slot for slot in slots if slot is not None]
    if isinstance(optimizer, tf.train.AdamOptimizer):
        slots.extend(optimizer._get_beta_accumulators())
    return tf.variables_initializer(slots, name=name)

def GetLogger(log_file_mode='w',log_file_path='log.txt'):
    log_format = logging.Formatter("%(asctime)s : %(message)s")

    # reset tensorflow handlers
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # register handlers
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file_path, mode=log_file_mode)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def GetDevices():
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    cpu = [x.name for x in local_device_protos if x.device_type == 'CPU'][0]
    devices = {'cpu':cpu, 'gpus':gpus}
    return devices

def CreateDir(dir):
    if not tf.gfile.Exists(dir):
        tf.gfile.MakeDirs(dir)
    return dir

def CreateOutputDir(prefix, filename):
    exp_id = os.path.splitext(os.path.basename(filename))[0]
    run_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return CreateDir(os.path.join(prefix, exp_id, run_name))

def GetAdamLR(adam):
    _beta1_power, _beta2_power = adam._get_beta_accumulators()
    current_lr = (adam._lr_t * tf.sqrt(1 - _beta2_power) / (1 - _beta1_power))
    return current_lr

def PrintParams(logger, params):
    logger.info({key: val for key, val in vars(params).items() if not isinstance(val, types.FunctionType)})


"""
1) BATCH_TYPE1: many images, many weights: 1-1
x[bs,...] 
w[bs,...]
 
2) BATCH_TYPE2: many images, many weights: each weight gets different batch of images 
x[nbs,ibs,...]
w[nbs,...]

3) BATCH_TYPE3: many images, many weights: all images per each weight
x[ibs,...]
w[nbs,...]

4) BATCH_TYPE4: many images, one weight
x[ibs,...]
w[...]
"""