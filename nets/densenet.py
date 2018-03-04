"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)

    if kernel_size[0] == 3:
        current = slim.conv2d(current, num_outputs, kernel_size, padding = 'same', scope=scope + '_conv')
    else:
        current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')

    current = slim.dropout(current, scope=scope + '_dropout')

    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def depth(num):
	return int(num)

def densenet(images, num_classes=1000, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24  # 12
    compression_rate = 0.5

    final_endpoint = 'Max_f'

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            #pass

            #print("densenet dense block layers:[6, 12, 32, 32]")
            #layers = [6, 12, 32, 32]
            layers = [6, 12, 12, 6]

            #imput 229 * 229 * 3

            #Convolution
            end_point = 'Conv2d_1a_7x7'
            net = slim.conv2d(images, 2 * growth, [7, 7], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            #Output Size 112 × 112 * growth * 2

            #Pooling
            end_point = 'MaxPool_2a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, padding = 'same', scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            #Output Size 56 * 56 * growth * 2

            #Dense Block1
            net = block(net, layers[0], growth, "Block_3a")
            #Output Size 56 * 56 * growth

            #Transition Layer
            end_point = 'Conv2d_4a_1x1'
            net = slim.conv2d(net, growth, [1, 1], stride=1, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            #Output Size 56 * 56 * growth

            end_point = 'AvgPool_5a_2x2'
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # Output Size 28 * 28 * growth

            #Dense Block2
            net = block(net, layers[1], growth, "Block_1b")
            #Output Size 28 * 28 * growth

            #Transition Layer
            end_point = 'Conv2d_2b_1x1'
            net = slim.conv2d(net, growth, [1, 1], stride=1, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            #Output Size 28 * 28 * growth

            end_point = 'AvgPool_3b_2x2'
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # Output Size 14 * 14 * growth

            #Dense Block3
            net = block(net, layers[2], growth, "Block_1c")
            #Output Size 14 * 14 * growth

            #Transition Layer
            end_point = 'Conv2d_2c_1x1'
            net = slim.conv2d(net, growth, [1, 1], stride=1, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            #Output Size 14 * 14 * growth

            end_point = 'AvgPool_3c_2x2'
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # Output Size 7 * 7 * growth

            # Dense Block4
            net = block(net, layers[3], growth, "Block_1d")
            # Output Size 7 * 7 * growth

            with tf.variable_scope('Logits'):
                end_point = 'GlobalAvgPool_1e_7x7'
                net = slim.avg_pool2d(net, [7, 7], stride=1, scope=end_point)
                end_points[end_point] = net

                end_point = 'PreLogitsFlatten'
                net = slim.flatten(net, end_point)
                end_points[end_point] = net

                if not num_classes:
                    return net, end_points
					
                end_point = 'Logits'
                logits = slim.fully_connected(net, num_classes, activation_fn=None, scope=end_point)		
                end_points[end_point] = logits
                end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')


    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
