# -*- coding: utf-8 -*-
"""
In-memory image passing support module
"""

from niftynet.io.image_source_base import ImageSourceBase

class MemoryImageSource(ImageSourceBase):
    """
    This class acts as a compatibility layer between a callback
    function yielding ID-data tuples and code expecting an ImageReader
    layer.
    """

    def __init__(self,
                 input_callback_functions,
                 num_subjects,
                 name='memory_image_source'):
        """
        :param input_callback_dicts: a dict of function, interpolation order
        tuples that for a given index an yield an image tensor and where
        the keys are the image collection names.
        :param num_subjects: number of subjects/defines the valid index range
        """

        super(MemoryImageSource, self).__init__()

        self._num_subjects = num_subjects
        self._input_callback_functions = input_callback_functions

    @property
    def names(self):
        return list(self._input_callback_functions.keys())

    @property
    def num_subjects(self):
        return self._num_subjects

    @property
    def _load_spatial_ranks(self):
        return {name: source(0) for name, source
                in self._input_callback_functions.items()}

    def get_image_index(self, subject_id):
        return int(subject_id)

    def _get_image_and_interp_dict(self, idx):
        image_data, interps = ({}, {})

        for name, funct in self._input_callback_functions:
            data, interp = funct(idx)
            image_data[name] = data
            interps[name] = interp

        return image_data, interps
