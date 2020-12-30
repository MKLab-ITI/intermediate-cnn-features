# Copyright 2018 Giorgos Kordopatis-Zilos. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Tensorflow implementation of the feature extraction process described in:

[1] Giorgos Kordopatis-Zilos, Symeon Papadopoulos, Ioannis Patras, Yiannis Kompatsiaris
    Near-Duplicate Video Retrieval by Aggregating Intermediate CNN Layers.
    International Conference on Multimedia Modeling (MMM), 2017.

This method is also used in:

[2] Giorgos Kordopatis-Zilos, Symeon Papadopoulos, Ioannis Patras, Yiannis Kompatsiaris
    Near-Duplicate Video Retrieval with Deep Metric Learning.
    IEEE International Conference on Computer Vision Workshop (ICCVW), 2017.
"""

from __future__ import division

import numpy as np
import tensorflow as tf

from future.utils import lrange


class CNN_tf():

    def __init__(self, name, model_ckpt):
        """
          Class initializer.

          Args:
            name: name of the CNN network
            model_ckpt: path to ckpt file of the pre-trained CNN model

          Raise:
            ValueError: if provided network name is not provided
        """
        self.net_name = name

        # intermediate convolutional layers to extract features
        if name == 'inception':
            from nets import inception_v4
            self.desired_size = inception_v4.inception_v4.default_image_size
            arg_scope = inception_v4.inception_v4_arg_scope()
            network = inception_v4.inception_v4
            preprocess = self.inc_preprocess
            self.layers = ['Mixed_3a', 'Mixed_4a', 'Mixed_5e', 'Mixed_6h', 'Mixed_7b']
        elif name == 'resnet':
            from nets import resnet_v1
            self.desired_size = resnet_v1.resnet_v1.default_image_size
            arg_scope = resnet_v1.resnet_arg_scope()
            network = resnet_v1.resnet_v1_152
            preprocess = self.vgg_preprocess
            self.layers = ['resnet_v1_152/block1', 'resnet_v1_152/block2',
                           'resnet_v1_152/block3', 'resnet_v1_152/block4']
        elif name == 'vgg':
            from nets import vgg
            self.desired_size = vgg.vgg_16.default_image_size
            arg_scope = vgg.vgg_arg_scope()
            network = vgg.vgg_16
            preprocess = self.vgg_preprocess
            self.layers = ['vgg_16/conv2/conv2_1', 'vgg_16/conv2/conv2_2',
                      'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2',
                      'vgg_16/conv3/conv3_3', 'vgg_16/conv4/conv4_1',
                      'vgg_16/conv4/conv4_2', 'vgg_16/conv4/conv4_3',
                      'vgg_16/conv5/conv5_1', 'vgg_16/conv5/conv5_2',
                      'vgg_16/conv5/conv5_3']
        else:
            raise ValueError('Network not found. Supported networks for Tensorflow framework: vgg, resnet, inception')

        self.input = tf.placeholder(tf.uint8,
                shape=(None, self.desired_size, self.desired_size, 3), name='input')
        vid_processed = preprocess(self.input)

        # create the CNN network
        with tf.contrib.slim.arg_scope(arg_scope):
            _, net = network(vid_processed, num_classes=None, is_training=False)

        # 1. normalize on channel dimension
        # 2. global max-pooling on channel dimension
        # 3. normalize feature vector
        net = [tf.nn.l2_normalize(tf.reduce_max(tf.nn.l2_normalize(tf.nn.relu(net[l])
                , 3, epsilon=1e-15), axis=(1, 2)), 1, epsilon=1e-15) for l in self.layers]

        self.output = tf.concat(net, axis=1)
        self.final_sz = self.output.get_shape()[1]

        init = self.load_model(model_ckpt)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def load_model(self, model_ckpt):
        """
          Function that loads the pre-trained model.

          Args:
            model_ckpt: path to ckpt file of the pre-trained CNN model

          Returns:
            tf_init: variables initializer
        """
        previous_variables = [var_name for var_name, _
                              in tf.contrib.framework.list_variables(model_ckpt)]
        restore_map = {variable.op.name: variable for variable in tf.global_variables()
                       if variable.op.name in previous_variables}
        tf.contrib.framework.init_from_checkpoint(model_ckpt, restore_map)
        tf_init = tf.global_variables_initializer()
        return tf_init

    def vgg_preprocess(self, images):
        """
          VGG preprocessing function applied on the provided
          images before they are fed to the network. It subtracts
          the ImageNet means from each colour channel.

          Args:
            images: tf tensor of input images to be processed

          Returns:
            images_pre: processed tf tensor of input images
        """
        images = tf.to_float(images)
        num_channels = images.get_shape().as_list()[-1]
        ax = images.get_shape().ndims - 1
        channels = tf.split(axis=ax, num_or_size_splits=num_channels, value=images)
        # subtract ImageNet means
        means = [122.68, 116.78, 103.94]
        for i in range(num_channels):
            channels[i] -= means[i]
        images_pre = tf.concat(axis=ax, values=channels)
        return images_pre

    def inc_preprocess(self, images):
        """
          Inception preprocessing function applied on the provided
          images before they are fed to the network.

          Args:
            images: tf tensor of input images to be processed

          Returns:
            images_pre: processed tf tensor of input images
        """
        images_pre = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images_pre = tf.subtract(images_pre, 0.5)
        images_pre = tf.multiply(images_pre, 2.0)
        return images_pre

    def extract(self, image_tensor, batch_sz):
        """
          Function that extracts intermediate CNN features for
          each input image.

          Args:
            image_tensor: numpy tensor of input images
            batch_sz: batch size

          Returns:
            features: extracted features from each input image
        """
        features = np.empty((image_tensor.shape[0], self.final_sz))
        for i in lrange(image_tensor.shape[0] // batch_sz + 1):
            batch = image_tensor[i * batch_sz:(i + 1) * batch_sz]
            if batch.size > 0:
                features[i * batch_sz:(i + 1) * batch_sz] = \
                    self.sess.run(self.output, feed_dict={self.input: batch})
        return features
