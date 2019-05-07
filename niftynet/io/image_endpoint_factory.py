# -*- coding: utf-8 -*-
"""
This module provides the factory for image data-set
partitioners, and image sources and sinks
"""

from niftynet.io.image_reader import ImageReader
from niftynet.io.image_writer import ImageWriter
from niftynet.io.memory_image_source import MemoryImageSource
from niftynet.io.memory_image_sink import MemoryImageSink
from niftynet.io.file_image_sets_partitioner import FileImageSetsPartitioner
from niftynet.io.memory_image_sets_partitioner import MemoryImageSetsPartitioner
from niftynet.utilities.decorators import singleton

# Name of the data_param namespace entry for the memory input sources
MEMORY_INPUT_CALLBACK_PARAM = 'input_callback_dicts'
# Name of the data_param namespace entry for number of in-memory subjects
MEMORY_INPUT_NUM_SUBJECTS_PARAM = 'num_subjects'
# Name of the data_param namespace entry for the output callback
MEMORY_OUTPUT_CALLBACK_PARAM = 'output_callback_function'
ENDPOINT_MEMORY = 'memory'
ENDPOINT_FILESYSTEM = 'filesystem'
# List of supported endpoints
SUPPORTED_ENDPOINTS = (ENDPOINT_FILESYSTEM, ENDPOINT_MEMORY)

@singleton
class ImageEndPointFactory(object):
    """
    This singleton class allows for the instantiation
    and configuration of any image source and/or data-set
    partitioner
    """

    _data_param = None
    _task_param = None
    _endpoint_type = ENDPOINT_FILESYSTEM
    _source_classes = {'filesystem': ImageReader,
                       'memory': MemoryImageSource}
    _sink_classes = {'filesystem': ImageWriter,
                     'memory': MemoryImageSink}
    _partitioner_classes = {'filesystem': FileImageSetsPartitioner,
                            'memory': MemoryImageSetsPartitioner}
    _partioner = None

    def set_params(self, data_param, task_param):
        """
        Configures this factory and determines which type of
        image endpoints and partitioners are instantiated.

        :param data_param: Data specification
        :param task_param: Application task specification
        """

        self._data_param = data_param
        self._task_param = task_param

        if not data_param is None \
           and MEMORY_OUTPUT_CALLBACK_PARAM in data_param \
           and MEMORY_INPUT_NUM_SUBJECTS_PARAM in data_param \
           and MEMORY_INPUT_CALLBACK_PARAM in data_param:
            self._endpoint_type = ENDPOINT_MEMORY
        else:
            self._endpoint_type = ENDPOINT_FILESYSTEM

    def create_partitioner(self):
        """
        Instantiates a new data-set partitioner suitable for the image
        end-point type specified via this factories parameters.
        """

        if self._data_param is None and self._task_param is None:
            raise RuntimeError('Application parameters must be set before any'
                               ' data set can be partitioned.')

        if self._partioner is None:
            self._partitioner = self._partitioner_classes[self._endpoint_type]()

        return self._partitioner

    def create_sources(self, dataset_names, phase, action):
        """
        Instantiates a list of sources for the specified application phase
        and data-set names.
        :param phase: an application life-cycle phase, e.g, TRAIN, VALID
        :param action: application action, e.g., TRAIN, INFER
        :param dataset_names: image collection/modality names
        :return: a configured image source
        """

        if self._partitioner is None:
            raise RuntimeError('Sources can only be instantiated after'
                               'data set partitioning')

        return [self._source_classes[self._endpoint_type](dataset_names)
                .initialise(self._data_param, self._task_param, subject_list)
                for subject_list in self._partitioner.get_image_lists_by(
                        phase=phase, action=action)]

