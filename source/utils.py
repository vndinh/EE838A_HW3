import tensorflow as tf
import os
import imageio
import numpy as np
from PIL import Image

from imresize import imresize
from config import config

# Parameters
scale = config.IMG.scale

def write_logs(filename, log, start=False):
  print(log)
  if start == True:
    f = open(filename, 'w')
    f.write(log + '\n')
  else:
    f = open(filename, 'a')
    f.write(log + '\n')
    f.close()

def get_filepath(path, is_blur, suffix):
  img_path = []
  directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
  directories = sorted(directories)
  for scene in directories:
    if is_blur:
      scene_path = os.path.join(path, scene, 'blur_gamma')
    else:
      scene_path = os.path.join(path, scene, 'sharp')
    for f in os.listdir(scene_path):
      if f.endswith(suffix):
        img_path.append(os.path.join(scene_path, f))

  img_path = sorted(img_path)
  return img_path

def img_read(img_dir):
  img = imageio.imread(img_dir, 'PNG-FI')
  h, w, _ = img.shape
  return img, h, w

def img_resize(img, output_shape):
  return imresize(img, scalar_scale=None, output_shape=output_shape)

def save_image(img, img_dir, id):
  _, h, w, c = img.shape
  img = np.reshape(img, [h,w,c])
  img = img * 255.0
  np.clip(img, 0, 255, out=img)
  img = img.astype('uint8')
  img_save = Image.fromarray(img)
  img_save.save(os.path.join(img_dir, '{:04d}.png'.format(id)))

def img_write(img_dir, img, fmt):
  _, h, w, c = img.shape
  img = np.clip(img, 0, 255)
  img = img.astype('uint8')
  img = np.reshape(img, [h,w,c])
  imageio.imwrite(img_dir, img, fmt)
