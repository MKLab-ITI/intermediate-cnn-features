import cv2
import numpy as np

def load_video(video, desired_size):
    try:
        frames = load_video_opencv(video, desired_size)
    except Exception as e:
        raise Exception('Can\'t load video {}\n{}'.format(video, e.message))

    return np.array(frames)

def load_image(image, desired_size):
    try:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if desired_size != 0:
            img = pad_and_resize(img, desired_size)
        return img
    except Exception as e:
        raise Exception('Can\'t load image {}\n{}'.format(image, e.message))

def load_video_opencv(video, desired_size):
    cap = cv2.VideoCapture(video)
    frames = []
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 29.97
    while cap.isOpened():
        ret, frame = cap.read()
        if isinstance(frame, np.ndarray):
            if int(count % round(fps)) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if desired_size != 0:
                    frame = pad_and_resize(frame, desired_size)
                frames.append(frame)
        else:
            break
        count += 1
    cap.release()
    return frames

def pad_and_resize(image, desired_size):
    old_size = image.shape[:2]
    ratio = float(desired_size) / max(old_size)
    img = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    new_size = img.shape[:2]
    delta_h = desired_size - new_size[0]
    delta_w = desired_size - new_size[1]
    top, bottom = delta_h / 2, delta_h - (delta_h / 2)
    left, right = delta_w / 2, delta_w - (delta_w / 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

