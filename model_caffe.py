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
Caffe implementation of the feature extraction process described in:

[1] Giorgos Kordopatis-Zilos, Symeon Papadopoulos, Ioannis Patras, Yiannis Kompatsiaris
    Near-Duplicate Video Retrieval by Aggregating Intermediate CNN Layers
    International Conference on Multimedia Modeling (MMM), 2017.

This method is also used in:

[2] Giorgos Kordopatis-Zilos, Symeon Papadopoulos, Ioannis Patras, Yiannis Kompatsiaris
    Near-Duplicate Video Retrieval with Deep Metric Learning.
    IEEE International Conference on Computer Vision Workshop (ICCVW), 2017.
"""

from __future__ import division

import caffe
import numpy as np

from future.utils import lrange

caffe.set_device(0)
caffe.set_mode_gpu()


class CNN_caffe():

    def __init__(self, name, prototxt, caffemodel):
        """
          Class initializer.

          Args:
            name: name of the CNN network
            prototxt: path to prototxt file of the pre-trained CNN model
            caffemodel: path to caffemodel file of the pre-trained CNN model

          Raise:
            ValueError: if provided network name is not provided
        """
        self.net_name = name

        # intermediate convolutional layers to extract features
        if name == 'googlenet':
            self.desired_size = 224
            self.layers = ['inception_3a/output', 'inception_3b/output',
                      'inception_4a/output', 'inception_4b/output',
                      'inception_4c/output', 'inception_4d/output',
                      'inception_4e/output', 'inception_5a/output',
                      'inception_5b/output']
        elif name == 'resnet':
            self.desired_size = 224
            self.layers = ['res2c', 'res3b7', 'res4b35', 'res5c']
        elif name == 'vgg':
            self.desired_size = 224
            self.layers = ['conv2_1', 'conv2_2', 'conv3_1',
                           'conv3_2', 'conv3_3', 'conv4_1',
                           'conv4_2', 'conv4_3', 'conv5_1',
                           'conv5_2', 'conv5_3']
        else:
            raise ValueError('Network not found. Supported networks for Caffe framework: googlenet, vgg, resnet')

        # load network
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        self.final_sz = np.sum(
            [self.net.blobs[layer].data.shape[1] for layer in self.layers])

    def preprocess(self, images):
        """
          Function that preprocess the provided images before
          they are fed to the network. It subtracts the ImageNet
          means from each colour channel and convert the shape of
          image input tensors.

          Args:
            images: numpy tensor of input images to be processed

          Returns:
            images_pre: processed numpy tensor of input images
        """
        images = images.astype(np.float32)
        num_channels = images.shape[-1]
        ax = len(images.shape) - 1
        # subtract ImageNet means
        channels = np.split(images, num_channels, axis=ax)
        means = [122.68, 116.67, 104.0]
        for i in range(num_channels):
            channels[i] -= means[i]
        images_pre = np.concatenate(channels, axis=ax)
        # swap colour channels
        images_pre = images_pre[:, :, :, ::-1]
        # transpose axis
        images_pre = images_pre.transpose((0, 3, 1, 2))
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
        preprocessed_images = self.preprocess(image_tensor)
        features = np.empty((preprocessed_images.shape[0], self.final_sz))
        for i in lrange(preprocessed_images.shape[0] // batch_sz + 1):
            batch = preprocessed_images[i * batch_sz:(i + 1) * batch_sz]
            if batch.size > 0:
                self.net.blobs['data'].reshape(*batch.shape)
                self.net.blobs['data'].data[...] = batch
                self.net.forward()
                start = 0
                for layer in self.layers:
                    activations = self.net.blobs[layer].data[...]
                    # normalize on channel dimension
                    activations /= np.linalg.norm(activations, axis=1, keepdims=True) + 1e-15
                    # global max-pooling on channel dimension
                    activations = activations.max(axis=(2, 3))
                    # normalize feature vector
                    activations /= np.linalg.norm(activations, axis=1, keepdims=True) + 1e-15
                    features[i * batch_sz:(i + 1) * batch_sz, start: start + activations.shape[1]] = activations
                    start += activations.shape[1]
        return features
