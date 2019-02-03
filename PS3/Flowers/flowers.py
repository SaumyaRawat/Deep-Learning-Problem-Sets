"""
From https://www.kaggle.com/alxmamaev/flowers-recognition
Preprocessing the Flowers dataset with torchvision
Output: center-cropped, 32x32 images
the .py files in the dandelion directory must be removed
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pdb

if __name__ == '__main__':
    img_path = 'flowers'
    out_path = '' # can also change to other paths such as 'flowers_npy'
    samples_path = 'flowers_smaples'
    img_size = 32
    samples_per_class = 2

    if (out_path is not '') and (not os.path.exists(out_path)):
        os.makedirs(out_path)

    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    tfms = transforms.Compose([transforms.Resize(img_size),
                               transforms.CenterCrop(img_size)])

    labels = sorted(os.listdir(img_path))
    imgs_npy, labels_npy = np.zeros((0,32,32,3), dtype=np.uint8), np.zeros((0,), dtype=np.int32)
    for nl, label in enumerate(labels):
        if label == '.DS_Store':
            continue
        subdir = os.path.join(img_path, label)
        imgs = sorted(os.listdir(subdir))
        print("Type: {}, label: {}, number of samples: {}".format(label, nl, len(imgs)))
        n_samples = 0
        for ni, img_name in enumerate(imgs):
            img_orig = Image.open(os.path.join(subdir, img_name))            
            img = np.array(tfms(img_orig))

            if n_samples < samples_per_class:
                # save some sample images
                img_orig.save(os.path.join(samples_path, '%s_%02d_original.jpg'%(label, n_samples)))
                Image.fromarray(img).save(os.path.join(samples_path, '%s_%02d_cropped.jpg'%(label, n_samples)))
                n_samples += 1

            imgs_npy = np.concatenate((imgs_npy, img[None,:,:,:]))
            labels_npy = np.append(labels_npy, nl)

    np.save(os.path.join(out_path, 'flower_imgs.npy'), imgs_npy)
    np.save(os.path.join(out_path, 'flower_labels.npy'), labels_npy)
    # randomly visualize one image
    # plt.imshow(imgs_npy[123])
    # plt.show()
