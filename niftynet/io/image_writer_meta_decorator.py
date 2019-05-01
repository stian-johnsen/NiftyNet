# -*- coding: utf-8 -*-
"""
This module provides classes that add additional
output meta-data to the F/S outputs of ImageWriter
"""
from __future__ import absolute_import

from niftynet.io.image_writer import ImageWriterBase


class ImageWriterDecorator(ImageWriterBase):
    """
    Base class for decorators on ImageWriter
    """

    def __init__(self,
                 base_writer):
        """
        :param base_writer: Underlying image writer instance
        """

        super(ImageWriterDecorator, self).__init__(
            base_writer.source, base_writer.interp_order)

        self.base_writer = base_writer

    @property
    def output_path(self):
        """
        Pass-through of underlying writer's output path.
        """

        return self.base_writer.output_path

    @property
    def postfix(self):
        """
        Pass-through of underlying writer's filename suffix.
        """

        return self.base_writer.postfix

    # pylint: disable=arguments-differ
    def layer_op(self, image_out, subject_id, image_in):
        self.base_writer(image_out, subject_id, image_in)
