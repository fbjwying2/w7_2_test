from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def preprocess_image(image, output_height, output_width, is_training):
    image = tf.to_float(image)
    image = tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
    return image