# -*- coding: utf-8 -*-
"""This module loads images from csv files and outputs numpy arrays."""
from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod, abstractproperty
import argparse
from copy import deepcopy
import numpy as np
import tensorflow as tf

from niftynet.io.misc_io import dtype_casting
from niftynet.layer.base_layer import Layer
from niftynet.utilities.user_parameters_helper import make_input_tuple
from niftynet.utilities.util_common import print_progress_bar, ParserNamespace
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.utilities.util_common import look_up_operations

DEFAULT_INTERP_ORDER = 1
SUPPORTED_DATA_SPEC = {
    'csv_file', 'path_to_search',
    'filename_contains', 'filename_not_contains', 'filename_removefromid',
    'interp_order', 'loader', 'pixdim', 'axcodes', 'spatial_window_size'}

def infer_tf_dtypes(image_array):
    """
    Choosing a suitable tf dtype based on the dtype of input numpy array.
    """
    return dtype_casting(
        image_array.dtype[0], image_array.interp_order[0], as_tf=True)


class ImageSourceBase(Layer):
    """
    Base class for F/S image reader and other image-input sources.
    """

    __metaclass__ = ABCMeta

    def __init__(self, name='image_source'):
        self._spatial_ranks = None
        self._shapes = None
        self._dtypes = None

        self.current_id = -1

        self.preprocessors = []
        super(ImageSourceBase, self).__init__(name='image_reader')

    @abstractmethod
    def _load_spatial_ranks(self):
        """
        :return: loads the spatial rank dict, returned by spatial_ranks
        """

        return

    @property
    def spatial_ranks(self):
        """
        :return: the shapes of the images in the collections provided
        by this source as dict of integers with image source names as keys.
        """

        if not self._spatial_ranks:
            self._spatial_ranks = self._load_spatial_ranks()

        return self._spatial_ranks

    @abstractmethod
    def _load_shapes(self):
        """
        :return: the dict of image shapes returned by shapes
        """

        return

    @property
    def shapes(self):
        """
        Image shapes before any preprocessing.

        :return: for every image source, the tuple of integers as image shape


        .. caution::

            To have fast access, the spatial dimensions are not accurate

                1. only read from the first image in list
                2. not considering effects of random augmentation layers
                    but time and modality dimensions should be correct
        """

        if not self._shapes:
            self._shapes = self._load_shapes()

        return self._shapes

    @abstractmethod
    def _load_dtypes(self):
        """
        :return: the dict of tensorflow data types returned by tf_dtypes
        """

        return

    @property
    def tf_dtypes(self):
        """
        Infer input data dtypes in TF
        (using the first image in the file list).
        """

        if not self._dtypes:
            self._dtypes = self._load_dtypes()

        return self._dtypes

    @abstractproperty
    def names(self):
        """
        :return: the list of input source names
        """

        return

    @abstractproperty
    def num_subjects(self):
        """
        :return the total number of subjects across the collections.
        """

        return

    @abstractmethod
    def get_subject_id(self, image_index):
        """
        Given an integer id returns the subject id.
        """

        return

    @abstractmethod
    def get_image_index(self, subject_id):
        """
        Given a subject id, return the file_list index
        :param subject_id: a string with the subject id
        :return: an int with the file list index
        """
        return

    @abstractmethod
    def _get_image_and_interp_dict(self, idx):
        """
        Given an index this function must produce two dictionaries
        containing one image data tensor and one interpolation
        order for every named image collection
        :return: one dictionary containing image data and one dictionary
        containing interpolation orders.
        """

        return

    # pylint: disable=arguments-differ,too-many-branches
    def layer_op(self, idx=None, shuffle=True):
        """
        this layer returns dictionaries::

            keys: self.output_fields
            values: image volume array

        """
        if idx is None:
            if shuffle:
                # training, with random list output
                idx = np.random.randint(self.num_subjects)
            else:
                # testing, with sequential output
                # accessing self.current_id, not suitable for multi-thread
                idx = self.current_id + 1
                self.current_id = idx

        image_data_dict, interp_order_dict \
            = self._get_image_and_interp_dict(idx)

        return idx, image_data_dict, interp_order_dict


def param_to_dict(input_data_param):
    """
    Validate the user input ``input_data_param``
    raise an error if it's invalid.

    :param input_data_param:
    :return: input data specifications as a nested dictionary
    """
    error_msg = 'Unknown ``data_param`` type. ' \
                'It should be a nested dictionary: '\
                '{"modality_name": {"input_property": value}} '\
                'or a dictionary of: {"modality_name": '\
                'niftynet.utilities.util_common.ParserNamespace}'
    data_param = deepcopy(input_data_param)
    if isinstance(data_param, (ParserNamespace, argparse.Namespace)):
        data_param = vars(data_param)
    if not isinstance(data_param, dict):
        raise ValueError(error_msg)
    for mod in data_param:
        mod_param = data_param[mod]
        if isinstance(mod_param, (ParserNamespace, argparse.Namespace)):
            dict_param = vars(mod_param)
        elif isinstance(mod_param, dict):
            dict_param = mod_param
        else:
            raise ValueError(error_msg)
        for data_key in dict_param:
            look_up_operations(data_key, SUPPORTED_DATA_SPEC)
        data_param[mod] = dict_param
    return data_param
