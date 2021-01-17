import numpy as np
import PIL.Image
from pathlib import Path


# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']


def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


def objective_L2(dst):
    dst.diff[:] = dst.data


def save_images(images_arr, last_image, output_dir="", fmt='jpeg'):
    '''
    Saving all images to disk
    '''
    print("Writing files to disk.")
    img_num = last_image - (len(images_arr) - 1)
    for image in images_arr:
        image = image / 255.0
        image = np.uint8(np.clip(image, 0, 1) * 255)
        PIL.Image.fromarray(image).save(Path(output_dir, str(img_num) + ".jpg"), fmt)
        img_num += 1

    return_image = images_arr[-1]
    return [return_image]