from __future__ import absolute_import, print_function

from niftynet.layer.base_layer import Layer
from niftynet.utilities.util_import import require_module

class RGBHistogramEquilisationLayer(Layer):
    """
    RGB histogram equilisation. Unlike the multi-modality general
    histogram normalisation this is done conventionally, on a
    per-image basis. This layer requires OpenCV.
    """

    def layer_op(self, image):
        """
        RGB histogram equilisation. Unlike the multi-modality general
        histogram normalisation this is done conventionally, on an
        individual image basis. This function requires OpenCV

        :param image: a 3-channel tensor
        :return: the equilised image
        """

        cv2 = require_module('cv2')

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[...,0] = cv2.equalizeHist(hsv_image[...,0])

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
