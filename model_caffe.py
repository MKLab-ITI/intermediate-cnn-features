import caffe
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()

class CNN_caffe():

    def __init__(self, name, prototxt, caffemodel):
        self.net_name = name
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
            raise ValueError('Network not found')

        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        self.final_sz = np.sum([self.net.blobs[layer].data.shape[1] for layer in self.layers])

    def preprocess(self, images):
        images = images.astype(np.float32)
        num_channels = images.shape[-1]
        ax = len(images.shape) - 1
        channels = np.split(images, num_channels, axis=ax)
        means = [122.68, 116.67, 104.0]
        for i in range(num_channels):
            channels[i] -= means[i]
        images = np.concatenate(channels, axis=ax)
        images = images[:, :, :, ::-1]
        images = images.transpose((0, 3, 1, 2))
        return images

    def extract(self, image_tensor, batch_sz):
        preprocessed_images = self.preprocess(image_tensor)
        features = np.empty((preprocessed_images.shape[0], self.final_sz))
        for i in xrange(preprocessed_images.shape[0] / batch_sz + 1):
            batch = preprocessed_images[i * batch_sz:(i + 1) * batch_sz]
            if batch.size > 0:
                self.net.blobs['data'].reshape(*batch.shape)
                self.net.blobs['data'].data[...] = batch
                self.net.forward()
                start = 0
                for layer in self.layers:
                    activations = self.net.blobs[layer].data[...]
                    activations /= np.linalg.norm(activations, axis=1, keepdims=True) + 1e-15
                    activations = activations.max(axis=(2, 3))
                    activations /= np.linalg.norm(activations, axis=1, keepdims=True) + 1e-15
                    features[i * batch_sz:(i + 1) * batch_sz, start: start + activations.shape[1]] = activations
                    start += activations.shape[1]
        return features
