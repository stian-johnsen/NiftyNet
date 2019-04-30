# -*- coding: utf-8 -*-
"""
Image output module
"""
from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

from niftynet.layer.base_layer import Layer

class ImageSinkBase(Layer):
    """
    Base class for passthrough layers that write images.
    """

    def __init__(self,
                 source,
                 interp_order,
                 name='image_sink'):
        """
        :param source: the image source of the input images
        for which this layer is to write the outputs.
        :param interp_order: polynomial order of the interpolation applied
        where needed on output.
        """

        super(ImageSinkBase, self).__init__(name=name)

        self._source = source
        self.interp_order = interp_order

    @property
    def source(self):
        """
        :return: the source of the images written by this layer
        """

        return self._source

    # pylint: disable=arguments-differ
    @abstractmethod
    def layer_op(self, image_data_out, image_id, image_data_in):
        """
        :param image_data_out: the voxel data to output
        :param image_id: the ID associated with the data
        :param image_data_in: the image object from which the output
        was generated
        """

        return
