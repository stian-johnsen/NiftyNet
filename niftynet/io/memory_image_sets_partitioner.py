# -*- coding: utf-8 -*-
"""
This module provides image set partitioning for
images kept in memory.
"""

from niftynet.io.image_sets_partitioner import BaseImageSetsPartitioner
from niftynet.utilities.decorators import singleton

@singleton
class MemoryImageSetsPartitioner(BaseImageSetsPartitioner):
    """
    Partitioning of images kept in RAM.
    """

    def initialise(self,
                   data_param,
                   new_partition=False,
                   data_split_file=None,
                   ratios=None):
        """
        :param num_subjects: number of subjects to partition
        :param new_partition: bool value indicating whether to generate new
            partition ids and overwrite csv file
            (this class will write partition file iff new_partition)
        :param data_split_file: location of the partition id file
        :param ratios: a tuple/list with two elements:
            ``(fraction of the validation set, fraction of the inference set)``
            initialise to None will disable data partitioning
            and get_file_list always returns all subjects.
        """

        self._num_subjects = data_param.num_subjects

    def get_file_lists_by(self, phase=ALL, action='train'):
        
