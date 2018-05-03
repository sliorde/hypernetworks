import numpy as np

from params import ResNetHyperParameters, DataParams


class ResnetWeights(dict):
    """
    This class holds all the trainable weights for a resnet networks, in a nested dictionary.
    for example, if `resnet_weights` is an instance of this class, you can use `resnet_weights['first_layer']['w']`, `resnet_weights['first_layer']['bn_offset']`, `resnet_weights['scales'][2]['blocks'][0]['conv_layers'][2]['w]'`, `resnet_weights['scales'][2]['blocks'][0]['projection']['bn_scale']`,`resnet_weights['final_layer']['b']` and so on
    """

    def __init__(self, hparams: ResNetHyperParameters=None, image_params: DataParams=None):
        super(ResnetWeights, self).__init__()

        self.treat_as_dict = False
        if (hparams is None) and (image_params is None):
            self.treat_as_dict = True
            return

        # the width and height of the feature image at the output of the first filter layer
        size = image_params.image_size / hparams.stride_between_scales

        # shape of filter and batch norm params for first filter layer in network
        w_shape = [int(hparams.first_layer_filter_size), int(hparams.first_layer_filter_size),
                   int(image_params.number_of_channels), int(hparams.first_layer_size)]
        bn_scale_shape = [int(size), int(size), int(hparams.first_layer_size)]
        bn_offset_shape = bn_scale_shape

        # this is the first filter layer in the network
        dict.__setitem__(self,'first_layer',WeightedLayer(w_shape=w_shape, bn_scale_shape=bn_scale_shape,bn_offset_shape=bn_offset_shape,name='first_layer'))

        # add scales to network
        dict.__setitem__(self,'scales',[])
        for scale_ind in range(len(hparams.num_blocks_per_scale)):
            self['scales'].append(ResnetScale(scale_ind, hparams, image_params,name='scale{:d}'.format(scale_ind)))

        # add final, fully connected layer to network. this layer does not have batch norm, and therefore does have a bias term
        w_shape = [
            int(hparams.first_layer_size * (hparams.stride_between_scales ** (len(hparams.num_blocks_per_scale) - 1))),
            int(image_params.num_classes)]
        b_shape = [int(image_params.num_classes)]
        dict.__setitem__(self, 'final_layer',WeightedLayer(w_shape=w_shape, b_shape=b_shape,name='final_layer'))

    def __setitem__(self, key, value):
        if self.treat_as_dict:
            dict.__setitem__(self, key, value)
        else:
            if key in self.keys():
                dict.__setitem__(self,key,value)
            else:
                raise KeyError()

    def CreateFeedDict(self,weights,with_filters: bool = True, with_batchnorm: bool = True):
        return {key:value for key,value in zip(self.AsList(with_filters=with_filters,with_batchnorm=with_batchnorm),weights.AsList(with_filters=with_filters,with_batchnorm=with_batchnorm)) if key is not None and value is not None}

    def AsList(self, shapes: bool = False, with_filters: bool = True, with_batchnorm: bool = True):
        """
        Args:
            shapes: if `True`, output will be a list of shapes. otherwise, it will be a list of weights (including biases and batchnorm)
            with_filters: if `True`, output will include the weights and biases of the weighted layers.
            with_batchnorm:  if `True`, output will include batchnorm params of the weighted layers.

        Returns:

        """
        weights = []
        weights.extend(self['first_layer'].AsList(shapes, with_filters, with_batchnorm))
        for a in self['scales']:
            weights.extend(a.AsList(shapes, with_filters, with_batchnorm))
        weights.extend(self['final_layer'].AsList(shapes, with_filters, with_batchnorm))
        return weights

    def NumberOfWeights(self, with_batchnorm: bool = True):
        """
        calculate number of weights in networks.
        Args:
            with_batchnorm:

        Returns:

        """
        shapes = self.AsList(shapes=True, with_batchnorm=with_batchnorm)
        return np.sum([np.prod(u) for u in shapes if u is not None])

    def WeightedLayerIterator(self):
        """
        return an iterator over all weighted layers in network.
        Returns:

        """
        scale = 0
        while scale <= len(self['scales']) + 1:
            if scale == 0:
                yield self['first_layer']
            elif scale == (len(self['scales']) + 1):
                yield self['final_layer']
            else:
                block = 0
                while block < len(self['scales'][scale - 1]['blocks']):
                    layer = 0
                    while layer < len(self['scales'][scale - 1]['blocks'][block]['conv_layers']):
                        yield self['scales'][scale - 1]['blocks'][block]['conv_layers'][layer]
                        layer += 1
                    if 'projection' in self['scales'][scale - 1]['blocks'][block].keys():
                        yield self['scales'][scale - 1]['blocks'][block]['projection']
                    block += 1
            scale += 1

    def AssignFromDictionary(self,d:dict):
        for key,value in d.items():
            if key == 'scales':
                for scale,new_scale in zip(self['scales'],value):
                    scale.AssignFromDictionary(new_scale)
            else:
                self[key].AssignFromDictionary(value)
        return self

class WeightedLayer(dict):
    def __init__(self, w_shape=None, b_shape=None, bn_scale_shape=None, bn_offset_shape=None,name=None):
        """
        represents a trainable layer (such as a filter or a fully connected layer), with weights, biases, and batch normalization parameters. The layer is initialized by giving the shapes for each of these. If a layer does not include some of these (e.g. no bias or no batchnorm), the shape should be `None`. All the trainable parameters are initialized to `None` and can later be given concrete values.
        Args:
            w_shape:
            b_shape:
            bn_scale_shape:
            bn_offset_shape:
        """
        super(WeightedLayer, self).__init__()

        self.treat_as_dict = False
        if (w_shape is None) and (b_shape is None) and (bn_scale_shape is None) and (bn_offset_shape is None) and (name is None):
            self.treat_as_dict = True
            return

        self.w_shape = w_shape
        if self.w_shape is not None:
            dict.__setitem__(self,'w',None)
        self.b_shape = b_shape
        if self.b_shape is not None:
            dict.__setitem__(self, 'b', None)
        self.bn_scale_shape = bn_scale_shape
        if self.bn_scale_shape is not None:
            dict.__setitem__(self, 'bn_scale', None)
        self.bn_offset_shape = bn_offset_shape
        if self.bn_offset_shape is not None:
            dict.__setitem__(self, 'bn_offset', None)

        self.name = name

    def __setitem__(self, key, value):
        if self.treat_as_dict:
            dict.__setitem__(self, key, value)
        else:
            if key in self.keys():
                dict.__setitem__(self,key,value)
            else:
                raise KeyError()

    def AssignFromDictionary(self,d:dict):
        for key,value in d.items():
            self[key] = value
        return self

    def AsList(self, shapes: bool = False, with_filters: bool = True, with_batchnorm: bool = True):
        l = []
        if shapes:
            if with_filters:
                l.extend([a for a in [self.w_shape, self.b_shape] if a is not None])
            if with_batchnorm:
                l.extend([a for a in [self.bn_scale_shape, self.bn_offset_shape] if a is not None])
        else:
            if with_filters:
                if 'w' in self.keys():
                    l.append(self['w'])
                if 'b' in self.keys():
                    l.append(self['b'])
            if with_batchnorm:
                if 'bn_scale' in self.keys():
                    l.append(self['bn_scale'])
                if 'bn_offset' in self.keys():
                    l.append(self['bn_offset'])
        return l

class ResnetScale(dict):
    def __init__(self, scale_ind:int=None, hparams:ResNetHyperParameters=None, image_params: DataParams=None, name=None):
        """
        represents a single "scale" of resnet. A scale is composed of all the layers between two pooling\downsampling opeartions. The actual learnable weights are all initialized to be `None`, and can later be modified.
        Args:
            scale_ind: the index of the scale. `0` indicate the first scale (right after the first pooling layer)
            hparams:
            image_params:
        """
        super(ResnetScale, self).__init__()

        self.treat_as_dict = False
        if (scale_ind is None) and (hparams is None) and (image_params is None) and (name is None):
            self.treat_as_dict = True
            return

        # add blocks to scale
        dict.__setitem__(self,'blocks',[])
        for block_ind in range(hparams.num_blocks_per_scale[scale_ind]):
            self['blocks'].append(ResnetBlock(block_ind, scale_ind, hparams, image_params,name=name+'_block{:d}'.format(block_ind)))
        self.name = name

    def __setitem__(self, key, value):
        if self.treat_as_dict:
            dict.__setitem__(self, key, value)
        else:
            if key in self.keys():
                dict.__setitem__(self,key,value)
            else:
                raise KeyError()

    def AssignFromDictionary(self,d:dict):
        for key,value in d.items():
            if key == 'blocks':
                for block,new_block in zip(self['blocks'],value):
                    block.AssignFromDictionary(new_block)
        return self

    def AsList(self, shapes: bool = False, with_filters: bool = True, with_batchnorm: bool = True):
        weights = []
        for a in self['blocks']:
            weights.extend(a.AsList(shapes, with_filters, with_batchnorm))
        return weights

class ResnetBlock(dict):
    def __init__(self, block_ind:int=None, scale_ind:int=None, hparams:ResNetHyperParameters=None,image_params:DataParams=None, name=None):
        """
        represents a single "block" of resnet. A block is composed of all layers, surrounded by a residual connection. We do not use bottleneck blocks. The actual learnable weights are all initialized to be `None`, and can later be modified.
        Args:
            block_ind: the index of the block withing a scale. `0` indicate the first block in a scale.
            scale_ind: the index of the scale. `0` indicate the first scale (right after the first pooling layer)
            hparams:
            image_params:
        """
        super(ResnetBlock,self).__init__()

        self.treat_as_dict = False
        if (block_ind is None) and (scale_ind is None) and (hparams is None) and (image_params is None) and (name is None):
            self.treat_as_dict = True
            return

        # calc the depth (input depth and output depth) of the filters for this scale
        depth = hparams.first_layer_size * (hparams.stride_between_scales ** scale_ind)

        # the first filter in the block will have different dimensions: its input depth will be smaller (and equal to the previous output depth). This input depth will also be used in residual connections that need projections.
        if scale_ind > 0 and block_ind == 0:
            prev_depth = hparams.first_layer_size * (hparams.stride_between_scales ** (scale_ind - 1))
        else:
            prev_depth = depth

        # the width and height of the feature image at the output of each filter
        size = image_params.image_size // (hparams.stride_between_scales ** (scale_ind + 2))

        # add projection layer (if needed)
        if (scale_ind > 0) and (block_ind == 0):
            w_shape = [1, 1, int(prev_depth), int(depth)]  # projection is a 1by1 filter
            bn_scale_shape = [int(size), int(size), int(depth)]
            bn_offset_shape = bn_scale_shape
            dict.__setitem__(self, 'projection', WeightedLayer(w_shape=w_shape, bn_scale_shape=bn_scale_shape,
                                            bn_offset_shape=bn_offset_shape,name=name+'_projection'))

        # add convolutional layers to block
        dict.__setitem__(self,'conv_layers',[])
        for k in range(hparams.layers_per_block):
            if k == 0:  # first layer in block might have different input depth
                w_shape = [int(hparams.filter_size), int(hparams.filter_size), int(prev_depth),
                           int(depth)]
            else:
                w_shape = [int(hparams.filter_size), int(hparams.filter_size), int(depth), int(depth)]
            bn_scale_shape = [int(size), int(size), int(depth)]
            bn_offset_shape = bn_scale_shape
            self['conv_layers'].append(WeightedLayer(w_shape=w_shape, bn_scale_shape=bn_scale_shape,
                                                  bn_offset_shape=bn_offset_shape,name=name+'_convlayer{:d}'.format(k)))
        self.name = name

    def __setitem__(self, key, value):
        if self.treat_as_dict:
            dict.__setitem__(self, key, value)
        else:
            if key in self.keys():
                dict.__setitem__(self,key,value)
            else:
                raise KeyError()

    def AssignFromDictionary(self,d:dict):
        for key,value in d.items():
            if key=='conv_layers':
                for conv_layer,new_conv_layer in zip(self['conv_layers'],value):
                    conv_layer.AssignFromDictionary(new_conv_layer)
            else:
                self[key].AssignFromDictionary(value)
        return self

    def AsList(self, shapes: bool = False, with_filters: bool = True, with_batchnorm: bool = True):
        weights = []
        if 'projection' in self.keys():
            weights.extend(self['projection'].AsList(shapes, with_filters, with_batchnorm))
        for l in self['conv_layers']:
            weights.extend(l.AsList(shapes, with_filters, with_batchnorm))
        return weights
