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
import os
import argparse
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
from utils import load_video, load_image


def feature_extraction_videos(model, cores, batch_sz, video_list, output_path):
    """
      Function that extracts the intermediate CNN features
      of each video in a provided video list.

      Args:
        model: CNN network
        cores: CPU cores for the parallel video loading
        batch_sz: batch size fed to the CNN network
        video_list: list of video to extract features
        output_path: path to store video features
    """
    video_list = {i: video.strip() for i, video in enumerate(open(video_list).readlines())}
    print '\nNumber of videos: ', len(video_list)
    print 'Storage directory: ', output_path
    print 'CPU cores: ', cores
    print 'Batch size: ', batch_sz

    print '\nFeature Extraction Process'
    print '=========================='
    pool = Pool(cores)
    future_videos = dict()
    output_list = []
    pbar = tqdm(xrange(np.max(video_list.keys())+1), mininterval=1.0, unit='video')
    for video in pbar:
        if os.path.exists(video_list[video]):
            video_name = os.path.splitext(os.path.basename(video_list[video]))[0]
            if video not in future_videos:
                video_tensor = load_video(video_list[video], model.desired_size)
            else:
                video_tensor = future_videos[video].get()
                del future_videos[video]

            # load videos in parallel
            for i in xrange(cores - len(future_videos)):
                next_video = np.max(future_videos.keys()) + 1 \
                    if len(future_videos) else video + 1

                if next_video in video_list and \
                    next_video not in future_videos and \
                        os.path.exists(video_list[next_video]):
                    future_videos[next_video] = pool.apply_async(load_video,
                        args=[video_list[next_video], model.desired_size])

            # extract features
            features = model.extract(video_tensor, batch_sz)

            path = os.path.join(output_path, '{}_{}'.format(video_name, model.net_name))
            output_list += ['{}\t{}'.format(video_name, path)]
            pbar.set_postfix(video=video_name)

            # save features
            np.save(path, features)
    np.savetxt('{}/video_feature_list.txt'.format(output_path), output_list, fmt='%s')


def feature_extraction_images(model, cores, batch_sz, image_list, output_path):
    """
      Function that extracts the intermediate CNN features
      of each image in a provided image list.

      Args:
        model: CNN network
        cores: CPU cores for the parallel video loading
        batch_sz: batch size fed to the CNN network
        image_list: list of image to extract features
        output_path: path to store video features
    """
    images = [image.strip() for image in open(image_list).readlines()]
    print '\nNumber of videos: ', len(image_list)
    print 'Storage directory: ', output_path
    print 'CPU cores: ', cores
    print 'Batch size: ', batch_sz

    print '\nFeature Extraction Process'
    print '=========================='
    pool = Pool(cores)
    batches = len(images)/batch_sz + 1
    features = []
    for batch in tqdm(xrange(batches), mininterval=1.0, unit='batches'):

        # load images in parallel
        future = []
        for image in images[batch * batch_sz: (batch+1) * batch_sz]:
            future += [pool.apply_async(load_image, args=[image, model.desired_size])]

        image_tensor = []
        for f in future:
            image_tensor += [f.get()]

        # extract features
        features += [model.extract(np.array(image_tensor), batch_sz)]

    # save features
    np.save(os.path.join(output_path, '{}_features'.format(model.net_name)), np.concatenate(features, axis=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str, required=True,
                        help='Name of the network')
    parser.add_argument('-f', '--framework', type=str, required=True,
                        help='Framework to be used. Options: \'caffe\' or \'tensorflow\'')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Output directory where the generated files will be stored')
    parser.add_argument('-v', '--video_list', type=str,
                        help='List of videos to extract features')
    parser.add_argument('-i', '--image_list', type=str,
                        help='List of images to extract features')
    parser.add_argument('-tf', '--tf_model', type=str,
                        help='Path to the .ckpt file of the pre-trained CNN model. '
                             'Required if Tensorflow framework is selected')
    parser.add_argument('-pt', '--prototxt', type=str,
                        help='Path to the .prototxt file of the pre-trained CNN model. '
                             'Required if Caffe framework is selected')
    parser.add_argument('-cm', '--caffemodel', type=str,
                        help='Path to the .caffemodel file of the pre-trained CNN model. '
                             'Required if Caffe framework is selected')
    parser.add_argument('-c', '--cores', type=int, default=8,
                        help='Number of CPU cores for the parallel load of images or video')
    parser.add_argument('-b', '--batch_sz', type=int, default=32,
                        help='Number of the images fed to the CNN network at once')
    args = vars(parser.parse_args())

    if args['framework'].lower() in ['tf', 'tensorflow']:
        print 'Selected framework: Tensorflow'
        if not args['tf_model']:
            raise Exception('--tf_model argument is not provided. It have to be provided when '
                            'Tensorflow framework is selected. Download: '
                            'https://github.com/tensorflow/models/tree/master/research/slim')
        elif '.ckpt' not in args['tf_model']:
            raise Exception('--tf_model argument is not a .ckpt file.')
        from model_tf import CNN_tf
        model = CNN_tf(args['network'].lower(), args['tf_model'])

    if args['framework'].lower() in ['cf', 'caffe']:
        print 'Selected framework: Caffe'
        if not args['prototxt'] or not args['caffemodel']:
            raise Exception('--prototxt or --caffemodel arguments are not provided. They have '
                            'to be provided when Caffe framework is selected. Download: '
                            'https://github.com/BVLC/caffe/wiki/Model-Zoo')
        elif '.prototxt' not in args['prototxt']:
            raise Exception('--prototxt is not a .prototxt file')
        elif '.caffemodel' not in args['caffemodel']:
            raise Exception('--caffemodel is not a .caffemodel file')

        from model_caffe import CNN_caffe
        model = CNN_caffe(args['network'].lower(), args['prototxt'], args['caffemodel'])

    print '\nCNN model has been built and initialized'
    print 'Architecture used: ', args['network']

    if args['video_list']:
        feature_extraction_videos(model, args['cores'], args['batch_sz'],
                                 args['video_list'], args['output_path'])
    elif args['image_list']:
        feature_extraction_images(model, args['cores'], args['batch_sz'],
                                 args['image_list'], args['output_path'])
    else:
        raise Exception('No --video_list or --image_list is given')