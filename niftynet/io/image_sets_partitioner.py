# -*- coding: utf-8 -*-
"""
This module manages a table of subject ids and
their associated image entries.
A subset of the table can be retrieved by partitioning the set of images into
subsets of ``Train``, ``Validation``, ``Inference``.
"""

from abc import ABCMeta, abstractmethod

from niftynet.engine.signal import TRAIN, VALID, INFER, ALL

SUPPORTED_PHASES = {TRAIN, VALID, INFER, ALL}


class BaseImageSetsPartitioner(object):
    """
    Base class for image sets partitioners
    """

    __metaclass__ = ABCMeta

    ratios = None
    data_param = None

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

        self.ratios = ratios
        self.data_param = data_param

    @abstractmethod
    def get_image_lists_by(self, phase=ALL, action='train'):
        """
        Get file lists by action and phase.

        This function returns file lists for training/validation/inference
        based on the phase or action specified by the user.

        ``phase`` has a higher priority:
        If `phase` specified, the function returns the corresponding
        file list (as a list).

        otherwise, the function checks ``action``:
        it returns train and validation file lists if it's training action,
        otherwise returns inference file list.

        :param action: an action
        :param phase: an element from ``{TRAIN, VALID, INFER, ALL}``
        :return:
        """

        return

    @abstractmethod
    def instantiate_source(self, phase=ALL):
        """
        Instantiates a list of image sources for the given
        phase in the application's life cycle.
        :return: fully configured image sources that can be used
        by applications for image input.
        """

        return

    @abstractmethod
    def has_phase(self, phase):
        """

        :return: True if the `phase` subset of images is not empty.
        """

        return

    @property
    def has_training(self):
        """

        :return: True if the TRAIN subset of images is not empty.
        """
        return self.has_phase(TRAIN)

    @property
    def has_inference(self):
        """

        :return: True if the INFER subset of images is not empty.
        """
        return self.has_phase(INFER)

    @property
    def has_validation(self):
        """

        :return: True if the VALID subset of images is not empty.
        """
        return self.has_phase(VALID)

    def reset(self):
        """
        reset all fields of this singleton class.
        """
