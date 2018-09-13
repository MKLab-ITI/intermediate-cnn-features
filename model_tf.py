import numpy as np
import tensorflow as tf

class CNN_tf():

    def __init__(self, name, model_ckpt):
        self.net_name = name
        if name == 'inception':
            from nets import inception_v4
            self.desired_size = inception_v4.inception_v4.default_image_size
            arg_scope = inception_v4.inception_v4_arg_scope()
            network = inception_v4.inception_v4
            model = model_ckpt
            preprocess = self.inc_preprocess
            self.layers = ['Mixed_3a', 'Mixed_4a', 'Mixed_5e', 'Mixed_6h', 'Mixed_7b']
        elif name == 'resnet':
            from nets import resnet_v1
            self.desired_size = resnet_v1.resnet_v1.default_image_size
            arg_scope = resnet_v1.resnet_arg_scope()
            network = resnet_v1.resnet_v1_152
            model = model_ckpt
            preprocess = self.vgg_preprocess
            self.layers = ['resnet_v1_152/block1', 'resnet_v1_152/block2',
                           'resnet_v1_152/block3', 'resnet_v1_152/block4']
        elif name == 'vgg':
            from nets import vgg
            self.desired_size = vgg.vgg_16.default_image_size
            arg_scope = vgg.vgg_arg_scope()
            network = vgg.vgg_16
            model = model_ckpt
            preprocess = self.vgg_preprocess
            self.layers = ['vgg_16/conv2/conv2_1', 'vgg_16/conv2/conv2_2',
                      'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2',
                      'vgg_16/conv3/conv3_3', 'vgg_16/conv4/conv4_1',
                      'vgg_16/conv4/conv4_2', 'vgg_16/conv4/conv4_3',
                      'vgg_16/conv5/conv5_1', 'vgg_16/conv5/conv5_2',
                      'vgg_16/conv5/conv5_3']
        else:
            raise ValueError('Network not found')

        self.input = tf.placeholder(tf.uint8, shape=(None, self.desired_size, self.desired_size, 3), name='input')
        vid_processed = preprocess(self.input)

        with tf.contrib.slim.arg_scope(arg_scope):
            _, net = network(vid_processed, num_classes=None, is_training=False)

        net = [tf.nn.l2_normalize(tf.reduce_max(tf.nn.l2_normalize(
            tf.nn.relu(net[l]), 3, epsilon=1e-15), axis=(1, 2)), 1, epsilon=1e-15) for l in self.layers]

        self.output = tf.concat(net, axis=1)
        self.final_sz = self.output.get_shape()[1]

        init = self.init_model(model)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def init_model(self, model):
        previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(model)]
        restore_map = {variable.op.name: variable for variable in tf.global_variables()
                       if variable.op.name in previous_variables}
        tf.contrib.framework.init_from_checkpoint(model, restore_map)
        return tf.global_variables_initializer()

    def vgg_preprocess(self, images):
        images = tf.to_float(images)
        num_channels = images.get_shape().as_list()[-1]
        ax = images.get_shape().ndims - 1
        channels = tf.split(axis=ax, num_or_size_splits=num_channels, value=images)
        means = [122.68, 116.78, 103.94]
        for i in range(num_channels):
            channels[i] -= means[i]
        images = tf.concat(axis=ax, values=channels)
        return images

    def inc_preprocess(self, images):
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)
        return images

    def extract(self, image_tensor, batch_sz):
        features = np.empty((image_tensor.shape[0], self.final_sz))
        for i in xrange(image_tensor.shape[0] / batch_sz + 1):
            batch = image_tensor[i * batch_sz:(i + 1) * batch_sz]
            if batch.size > 0:
                features[i * batch_sz:(i + 1) * batch_sz] = \
                    self.sess.run(self.output, feed_dict={self.input: batch})
        return features
