import os
from glob import glob

import numpy as np
from PIL import Image


def load_dataset(rel_path='.', mode="training", resize=False, resize_shape=(256, 256)):
    if os.path.exists("storage/datasets/{}/image.npy".format(mode)) and \
            os.path.exists("storage/datasets/{}/label.npy".format(mode)) and \
            os.path.exists("storage/datasets/{}/mask.npy".format(mode)):
        new_input_tensor = np.load("storage/datasets/{}/image.npy".format(mode))
        new_label_tensor = np.load("storage/datasets/{}/label.npy".format(mode))
        new_mask_tensor = np.load("storage/datasets/{}/mask.npy".format(mode))
        return new_input_tensor, new_label_tensor, new_mask_tensor

    train_image_files = sorted(glob(os.path.join(rel_path, 'storage/DRIVE/{}/images/*.tif'.format(mode))))
    train_label_files = sorted(glob(os.path.join(rel_path, 'storage/DRIVE/{}/1st_manual/*.gif'.format(mode))))
    train_mask_files = sorted(glob(os.path.join(rel_path, 'storage/DRIVE/{}/mask/*.gif'.format(mode))))

    for i, filename in enumerate(train_image_files):
        print('adding {}th {} image : {}'.format(i + 1, mode, filename))
        image = Image.open(filename)
        if not resize:
            pass
        else:
            image = image.resize(resize_shape, Image.ANTIALIAS)
        imagemat = np.array(image).astype('float')
        imagemat = imagemat / 255.0
        if i != 0:
            input_tensor = np.concatenate((input_tensor, np.expand_dims(imagemat, axis=0)), axis=0)
        else:
            input_tensor = np.expand_dims(imagemat, axis=0)
    new_input_tensor = np.moveaxis(input_tensor, 3, 1)

    for i, filename in enumerate(train_label_files):
        print('adding {}th {} label : {}'.format(i + 1, mode, filename))
        Image_label = Image.open(filename)
        if not resize:
            pass
        else:
            Image_label = Image_label.resize(resize_shape, Image.ANTIALIAS)
            Image_label = Image_label.convert('1')
        label = np.array(Image_label)
        label = label / 1.0
        if i != 0:
            label_tensor = np.concatenate((label_tensor, np.expand_dims(label, axis=0)), axis=0)
        else:
            label_tensor = np.expand_dims(label, axis=0)
    new_label_tensor = np.stack((label_tensor[:, :, :], 1 - label_tensor[:, :, :]), axis=1)

    for i, filename in enumerate(train_mask_files):
        print('adding {}th {} mask : {}'.format(i + 1, mode, filename))
        Image_mask = Image.open(filename)
        if not resize:
            pass
        else:
            Image_mask = Image_mask.resize(resize_shape, Image.ANTIALIAS)
            Image_mask = Image_mask.convert('1')
        mask = np.array(Image_mask)
        mask = mask / 1.0
        if i != 0:
            mask_tensor = np.concatenate((mask_tensor, np.expand_dims(mask, axis=0)), axis=0)
        else:
            mask_tensor = np.expand_dims(mask, axis=0)
    new_mask_tensor = np.stack((mask_tensor[:, :, :], mask_tensor[:, :, :]), axis=1)

    np.save("storage/datasets/{}/image.npy".format(mode), new_input_tensor)
    np.save("storage/datasets/{}/label.npy".format(mode), new_label_tensor)
    np.save("storage/datasets/{}/mask.npy".format(mode), new_mask_tensor)

    return new_input_tensor, new_label_tensor, new_mask_tensor
