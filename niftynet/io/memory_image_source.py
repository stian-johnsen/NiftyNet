# -*- coding: utf-8 -*-
"""
In-memory image passing support module
"""

import numpy as np

from niftynet.io.image_source_base import ImageSourceBase

class MemoryImageSource(ImageSourceBase):
    """
    This class acts as a compatibility layer between a callback
    function yielding ID-data tuples and code expecting an ImageReader
    layer.
    """

    def __init__(self,
                 function_names,
                 name='memory_image_source'):
        """
        :param input_callback_dicts: a dict of function, interpolation order
        tuples that for a given index an yield an image tensor and where
        the keys are the image collection names.
        :param num_subjects: number of subjects/defines the valid index range
        """

        super(MemoryImageSource, self).__init__()

        self._total_num_subjects = 0
        self._input_callback_functions = None
        self._phase_indices = None
        self._function_names = function_names

    def initialise(self,
                   data_param,
                   task_param,
                   phase_indices):
        """
        :param data_param: Data specification
        :param task_param: Application task specification
        :param phase_indices: subset of image indices to consider in this phase
        :return: self
        """

        self._total_num_subjects = data_param.num_subjects
        self._input_callback_functions\
            = {name: data_param.input_callback_dicts[name]
               for name in self._function_names}
        self._phase_indices = phase_indices

        return self

    @property
    def names(self):
        return list(self._input_callback_functions.keys())

    @property
    def num_subjects(self):
        return len(self._phase_indices)

    def _load_spatial_ranks(self):
        return {name: 2 if source(0)[0].shape[2] <= 1 else 3
                for name, source in self._input_callback_functions.items()}

    def _load_shapes(self):
        return {name: source(0)[0].shape for name, source
                in self._input_callback_functions.items()}

    def _load_dtypes(self):
        return {name: source(0)[0].dtype for name, source
                in self._input_callback_functions.items()}

    def get_image_index(self, subject_id):
        idx = np.argwhere(np.array(self._phase_indices) == int(subject_id))

        return idx[0,0] if idx else -1

    def get_subject_id(self, image_index):
        return str(self._phase_indices[image_index])

    def _get_image_and_interp_dict(self, idx):
        image_data, interps = ({}, {})

        for name, funct in self._input_callback_functions:
            data, interp = funct(self._phase_indices[idx])
            image_data[name] = data
            interps[name] = interp

        return image_data, interps
