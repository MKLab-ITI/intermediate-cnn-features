import os
import argparse
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
from utils import load_video, load_image

def feature_extraction_videos(model, cores, batch_sz, video_list, output_path):
    video_dirs = {i: video.strip() for i, video in enumerate(open(video_list).readlines())}

    pool = Pool(cores)
    future_videos = dict()
    pbar = tqdm(xrange(np.max(video_dirs.keys())+1), mininterval=1.0, unit='video')
    for video in pbar:
        if os.path.exists(video_dirs[video]):
            video_name = os.path.splitext(os.path.basename(video_dirs[video]))[0]
            if video not in future_videos:
                video_tensor = load_video(video_dirs[video], model.desired_size)
            else:
                video_tensor = future_videos[video].get()
                del future_videos[video]

            for i in xrange(cores - len(future_videos)):
                if len(future_videos):
                    next_video = np.max(future_videos.keys()) + 1
                else:
                    next_video = video + 1

                if next_video in video_dirs and \
                    next_video not in future_videos and \
                        os.path.exists(video_dirs[next_video]):
                    future_videos[next_video] = pool.apply_async(load_video,
                        args=[video_dirs[next_video], model.desired_size])

            features = model.extract(video_tensor, batch_sz)

            np.save(os.path.join(output_path, video_name), features)
            pbar.set_postfix(video=video_name)

def feature_extraction_images(model, cores, batch_sz, image_list, output_path):
    images = [image.strip() for image in open(image_list).readlines()]

    pool = Pool(cores)
    batches = len(images)/batch_sz + 1
    features = []
    for batch in tqdm(xrange(batches), mininterval=1.0, unit='batches'):
        future = []
        for image in images[batch * batch_sz: (batch+1) * batch_sz]:
            future += [pool.apply_async(load_image, args=[image, model.desired_size])]

        image_tensor = []
        for f in future:
            image_tensor += [f.get()]

        features += [model.extract(np.array(image_tensor), batch_sz)]

    np.save(os.path.join(output_path, 'intermediate_cnn'), np.concatenate(features, axis=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str, required=True)
    parser.add_argument('-f', '--framework', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('-v', '--video_list', type=str)
    parser.add_argument('-i', '--image_list', type=str)
    parser.add_argument('-tf', '--tf_model', type=str)
    parser.add_argument('-pt', '--prototxt', type=str)
    parser.add_argument('-cm', '--caffemodel', type=str)
    parser.add_argument('-c', '--cores', type=int, default=8)
    parser.add_argument('-b', '--batch_sz', type=int, default=32)
    args = vars(parser.parse_args())

    if args['framework'] == 'tf' or \
        args['framework'] == 'tensorflow':
        if not args['tf_model']:
            raise Exception('Tensorflow slim model not given')
        elif '.ckpt' not in args['tf_model']:
            raise Exception('Tensorflow slim model is not a .ckpt file')
        from model_tf import CNN_tf
        model = CNN_tf(args['network'], args['tf_model'])

    if args['framework'] == 'caffe':
        if not args['prototxt'] or not args['caffemodel']:
            raise Exception('prototxt or caffemodel not given')
        elif '.prototxt' not in args['prototxt']:
            raise Exception('prototxt is not a .prototxt file')
        elif '.caffemodel' not in args['caffemodel']:
            raise Exception('caffemodel is not a .caffemodel file')
        from model_caffe import CNN_caffe
        model = CNN_caffe(args['network'], args['prototxt'], args['caffemodel'])

    if args['video_list']:
        feature_extraction_videos(model, args['cores'], args['batch_sz'],
                                 args['video_list'], args['output_path'])
    elif args['image_list']:
        feature_extraction_images(model, args['cores'], args['batch_sz'],
                                 args['image_list'], args['output_path'])
    else:
        raise Exception('No image or video list is given')