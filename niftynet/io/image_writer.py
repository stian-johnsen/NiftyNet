# -*- coding: utf-8 -*-
"""
Image F/S output module
"""
from __future__ import absolute_import

import os.path

from niftynet.io.image_sink_base import ImageSinkBase
import niftynet.io.misc_io as misc_io

class ImageWriter(ImageSinkBase):
    """
    F/S output image writer class
    """

    def __init__(self,
                 source,
                 interp_order,
                 output_path='.',
                 postfix='_niftynet_out'):
        """
        :param output_path: output directory
        :param postfix: filename postfix applied to images
        """

        super(ImageWriter, self).__init__(source, interp_order)

        self.output_path = os.path.abspath(output_path)
        self.postfix = postfix
        self.inferred_cleared = False

    def layer_op(self, image_data_out, image_id, image_data_in):
        subject_name = self.source.get_subject_id(image_id)
        filename = "{}{}.nii.gz".format(subject_name, self.postfix)
        misc_io.save_data_array(self.output_path,
                                filename,
                                image_data_out,
                                image_data_in,
                                self.interp_order)
        self.log_inferred(subject_name, filename)

    def log_inferred(self, subject_name, filename):
        """
        This function writes out a csv of inferred files

        :param subject_name: subject name corresponding to output
        :param filename: filename of output
        :return:
        """
        inferred_csv = os.path.join(self.output_path, 'inferred.csv')
        if not self.inferred_cleared:
            if os.path.exists(inferred_csv):
                os.remove(inferred_csv)
            self.inferred_cleared = True
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
        with open(inferred_csv, 'a+') as csv_file:
            filename = os.path.join(self.output_path, filename)
            csv_file.write('{},{}\n'.format(subject_name, filename))

