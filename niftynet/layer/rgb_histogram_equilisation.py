from __future__ import absolute_import, print_function

import numpy as np
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
        :param image: a 3-channel tensor assumed to be an image in floating-point
        RGB format (each channel in [0, 1])
        :return: the equilised image
        """

        cv2 = require_module('cv2')

        image = image[...,::-1]
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        intensity = (255*yuv_image[...,0]).astype(np.uint8)
        yuv_image[...,0] = cv2.equalizeHist(intensity).astype(np.float32)/255

        return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)[...,::-1]
