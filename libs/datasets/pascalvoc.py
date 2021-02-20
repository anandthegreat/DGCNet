import os
import tarfile
import collections
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import cv2
import random
import numpy as np
import sys

DATASET_YEAR_DICT = {
    '2012': {
        'filename': 'VOCtrainval_11-May-2012.tar',
        'base_dir': os.path.join('VOCdevkit', 'VOC2012')
    }
}

# train distribution [182871358, 2031356, 713150, 4989269, 2301218, 1463550, 2336871, 3844779, 
# 0, 0, 0, 0, 0, 0, 0, 0, 1428974, 2114231, 3422716, 0, 2561556, 14791372]
# test distribution [180645590, 1451179, 821086, 5017211, 2600016, 2103955, 2416598, 3791700, 
# 0, 0, 0, 0, 0, 0, 0, 0, 1220470, 1739211, 2989059, 0, 2196530, 15573795]


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def build_colormap2label():
    """Build an RGB color to label mapping for segmentation."""
    colormap2label = np.zeros(256 ** 3)   #create an array of size 16777216 (=256^3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256 + colormap[1])*256 + colormap[2]] = i    #hash(label value) = label
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map an RGB color to a label."""
    colormap = colormap.astype(np.int32)
    # print("LABEL'S SHAPE IN voc_label_indices IS: ", colormap.shape)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])     
    #calculate label value and find it's label mapping from colormap2label
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """Randomly crop for both feature and label images."""
    feature = transforms.ToTensor(feature)
    label = transforms.ToTensor(label)
    rect = transforms.RandomCrop.get_params(feature, (height, width))
    feature = transforms.functional.crop(feature, *rect)
    label = transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root,
            year= "2012",
            max_iters=None,
            crop_size=(320,480),
            mean=(128,128,128),
            vars=(1,1,1),
            scale=False,
            ignore_label=255,
            image_set= "train",
            download= False,
            transform= None,
            target_transform = None,
            transforms = None,
    ):
#        super(VOCSegmentation, self).__init__(root, transforms, transform, target_transform)
        self.scale = scale
        self.crop_h, self.crop_w = crop_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.vars = vars
        self.year = year
        self.filename = DATASET_YEAR_DICT[year]['filename']
        # valid_sets = ["train", "trainval", "val"]
        self.image_set = image_set
        self.root = root
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')
        self.transforms = transforms
        self.colormap2label = build_colormap2label()

        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted')

        splits_dir = os.path.join(root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        if not max_iters==None:
            self.images = self.images * int(np.ceil(float(max_iters) / len(self.images)))
            self.masks = self.masks * int(np.ceil(float(max_iters) / len(self.masks)))
        # print("first image file name: ", self.images[0])
        # print("first mask file name: ", self.masks[0])
        assert (len(self.images) == len(self.masks))

    # As some images in the dataset may be smaller than the output dimensions specified 
    # for random cropping, we must remove these example by using a custom filter function.
    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        label = cv2.imread(self.masks[index])
        # print("LABEL'S SHAPE IS: ", label.shape)
        # label = voc_label_indices(label, self.colormap2label)
        # print("successfully loaded image and label")
        image = self.filter(image)
        label = self.filter(label)
        image, label = voc_rand_crop(image, label, *self.crop_size)
        return image, voc_label_indices(label, self.colormap2label)

        # if self.transforms is not None:
        #     image, label = self.transforms(image, label)
        # if self.scale:
        #     f_scale = 0.7 + random.randint(0, 14) / 10.0
        #     image=cv2.resize(image,None,fx=f_scale,fy=f_scale,interpolation=cv2.INTER_LINEAR)
        #     label=cv2.resize(label,None,fx=f_scale,fy=f_scale,interpolation=cv2.INTER_NEAREST)
        # image = np.asarray(image, np.float32)
        # image -= self.mean
        # image /= self.vars
        # img_h, img_w = label.shape
        # pad_h = max(self.crop_h - img_h, 0)
        # pad_w = max(self.crop_w - img_w, 0)
        # if pad_h > 0 or pad_w > 0:
        #     img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
        #         pad_w, cv2.BORDER_CONSTANT,
        #         value=(0.0, 0.0, 0.0))
        #     label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
        #         pad_w, cv2.BORDER_CONSTANT,
        #         value=(self.ignore_label,))
        # else:
        #     img_pad, label_pad = image, label

        # img_h, img_w = label_pad.shape
        # h_off = random.randint(0, img_h - self.crop_h)
        # w_off = random.randint(0, img_w - self.crop_w)

        # image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        # label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)

        # image = image.transpose((2, 0, 1))
        # print(image.shape, label.shape)
        # return image, label


    def __len__(self):
        return len(self.images)

