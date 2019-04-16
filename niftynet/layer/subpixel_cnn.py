# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer

class SubPixelCNNLayer(TrainableLayer):
    """
    Implementation of Shi et al.'s sub-pixel CNN single-image
    upsampling method.

    Based on Shi et al.: "Real-Time Single Image and Video
    Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural
    Network"
    """

    def __init__(self,
                 upsample_factor=3,
                 layer_configurations=((5, 64),
                                       (3, 32),
                                       (3, -1)),
                 acti_func='tanh',
                 with_bn=True,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='subpixel_cnn'):
        """
        :param upsample_factor: zoom-factor/image magnification factor
        :param layer_configurations: N pairs consisting of a kernel size and
        a feature-map size, where N is the number of layers in the net. The last
        layer must have a feature-map size of -1.
        """

        super(SubPixelCNNLayer, self).__init__(name=name)

        if layer_configurations[-1][1] != -1:
            raise ValueError('The size of the last feature map must be -1')

        self.upsample_factor = upsample_factor
        self.layer_configurations = layer_configurations
        self.acti_func = acti_func

        self.base_layer_params = {'with_bias': True,
                                  'with_bn': with_bn,
                                  'w_initializer': w_initializer,
                                  'b_initializer': b_initializer,
                                  'w_regularizer': w_regularizer,
                                  'b_regularizer': b_regularizer}

    def layer_op(self, lr_images, is_training=True, keep_prob=1.0):
        input_shape = lr_images.shape.as_list()
        batch_size = input_shape[0]
        input_shape = input_shape[1:]
        nof_dims = len(input_shape) - 1

        if any(i is None or i <= 0 for i in input_shape):
            raise ValueError('The image shape must be known in advance.')

        nof_channels = input_shape[-1]

        features = lr_images
        for i, (ksize, nof_features) in enumerate(self.layer_configurations):
            name = 'fmap_{}'.format(i)

            if nof_features > 0:
                conv = ConvolutionalLayer(nof_features,
                                          kernel_size=ksize,
                                          acti_func=self.acti_func,
                                          name=name,
                                          **self.base_layer_params)
            else:
                nof_features = nof_channels*(self.upsample_factor**nof_dims)
                conv = ConvolutionalLayer(nof_features,
                                          kernel_size=ksize,
                                          acti_func=None,
                                          name=name,
                                          **self.base_layer_params)

            features = conv(features, is_training=is_training,
                            keep_prob=keep_prob)

        output_shape = [batch_size] \
            + [self.upsample_factor*i for i in input_shape[:-1]] \
            + [nof_channels]

        print('in shape = ', lr_images.shape.as_list())
        print('out ftrs = ', features.shape.as_list())
        print('out shape = ', output_shape)

        shuffled = tf.contrib.periodic_resample.periodic_resample(features,
                                                                  output_shape,
                                                                  name='shuffle')

        print('shuffled = ', shuffled)

        return shuffled
