import cv2
import os
import argparse
import glob
from PIL import Image, ImageOps
import numpy as np
import shutil
import random

parser = argparse.ArgumentParser(description='video2dataset')
parser.add_argument('arg', help='movie name')
args = parser.parse_args()

width = 128
height = 128


def save_frames(video_path, dir_path, basename, ext='jpg'):
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n), ext), frame)
            n += 1
        else:
            return

def image_resize(width, height):
    files = glob.glob('data/img/*.jpg')

    for f in files:
      img = Image.open(f)
      img_resize = img.resize((width, height), Image.BOX)
      title, ext = os.path.splitext(f)
      img_resize.save(title + ext)

# only mirror
def data_augument():
    files = glob.glob('data/img/*.jpg')

    n = 0
    file_size = len(files)
    for f in files:
      img = Image.open(f)
      img_mirror = ImageOps.mirror(img)
      img_mirror.save('data/img/img_' + str(int(file_size + n)) + '.jpg', quality=95)
      n += 1


save_frames('data/video/' + args.arg, 'data/img/', 'img')
image_resize(width, height)
data_augument()