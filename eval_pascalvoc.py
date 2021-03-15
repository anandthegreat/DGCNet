import argparse
from scipy import ndimage
import numpy as np
import json
from Voc_loader import VOCLoader
import torch
from torch.utils import data
import torch.nn as nn
import os
import sys
from tqdm import tqdm
from metrics import custom_conf_matrix
from math import ceil
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import scipy.misc as m
from PIL import Image
from tqdm import tqdm

DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './data/cityscapes/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500  # Number of images in the validation set.
INPUT_SIZE = 832
RESTORE_FROM = './deeplab_resnet.pth'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data_set", type=str, default="cityscapes", help="dataset to train")
    parser.add_argument("--arch",type=str,default="CascadeRelatioNet_res50")
    parser.add_argument("--num_classes", type=int, default=21,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore models parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--rgb", type=int, default=0)
    return parser.parse_args()


def val():
    """Create the models and start the evaluation process."""
    args = get_arguments()
    train_loader = data.DataLoader(
        VOCLoader(args.data_dir,do_transform=True),
        shuffle=True,
        batch_size=1,
    ) 
    val_loader = data.DataLoader(
        VOCLoader(args.data_dir,portion="val",do_transform=True),
        batch_size=1,
    ) 
    
    h, w = args.input_size, args.input_size
    input_size = (h, w)
    import libs.models as models
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict,strict=False)

    model.eval()
    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_mat = custom_conf_matrix([i for i in range(0,21)],21)
    with torch.no_grad():
        val_loss=0
        for vi, (vimg,vlbl) in enumerate(tqdm(val_loader)):
            vimg, vlbl = vimg.to(device), vlbl.to(device)
            vout = model(vimg)
            print("------vout------")
            print(len(vout))
            # pred = vout.data.max(1)[1].cpu().numpy()
            gt = vlbl.data.cpu().numpy()
            print("------gt------")
            print(gt.shape)
            sys.exit()
            # conf_mat.update_step(gt.flatten(), pred.flatten())  
    
    score = conf_mat.compute_mean_iou() 
    print("mean iou ",score)
    # conf_mat.reset()        


if __name__ == '__main__':
    val()