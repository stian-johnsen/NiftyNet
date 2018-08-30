# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_OP = set(['AVG', 'MAX'])


class SELayer(Layer):
    def __init__(self,
                 func='AVG',
                 reduction_ratio=16,
                 name='squeeze_excitation'):
        self.func = func.upper()
        self.reduction_ratio = reduction_ratio
        self.layer_name = '{}_{}'.format(self.func.lower(), name)
        super(SELayer, self).__init__(name=self.layer_name)

        look_up_operations(self.func, SUPPORTED_OP)

    def layer_op(self, input_tensor):
        # squeeze: global information embedding
        input_rank = len(input_tensor.shape)
        reduce_indices = list(range(input_rank))[1:-1]
        if self.func == 'AVG':
            squeeze_tensor = tf.reduce_mean(input_tensor, axis=reduce_indices)
        elif self.func == 'MAX':
            squeeze_tensor = tf.reduce_max(input_tensor, axis=reduce_indices)
        else:
            raise NotImplementedError("pooling function not supported")

        # excitation: adaptive recalibration
        num_channels = int(squeeze_tensor.shape[-1])
        reduction_ratio = self.reduction_ratio
        if num_channels % reduction_ratio != 0:
            raise ValueError(
                "reduction ratio incompatible with "
                "number of input tensor channels")

        num_channels_reduced = num_channels / reduction_ratio
        fc1 = FullyConnectedLayer(num_channels_reduced,
                                  with_bias=False,
                                  with_bn=False,
                                  acti_func='relu',
                                  name='se_fc_1')
        fc2 = FullyConnectedLayer(num_channels,
                                  with_bias=False,
                                  with_bn=False,
                                  acti_func='sigmoid',
                                  name='se_fc_2')

        fc_out_1 = fc1(squeeze_tensor)
        fc_out_2 = fc2(fc_out_1)

        while len(fc_out_2.shape) < input_rank:
            fc_out_2 = tf.expand_dims(fc_out_2, axis=1)

        output_tensor = tf.multiply(input_tensor, fc_out_2)

        return output_tensor
