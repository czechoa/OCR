# from datetime import time
import time
import numpy as np
import torch

from utils.detect.craft_pytorch import imgproc
from utils.detect.craft_pytorch.craft import CRAFT
from utils.detect.craft_pytorch.test_net import test_net, copyStateDict


trained_model='saved_models/craft_mlt_25k.pth'

net = CRAFT()  # initialize

net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

net.eval()


def craft_main(image_path, trained_model='saved_models/craft_mlt_25k.pth', text_threshold=0.7, low_text=0.4,
               link_threshold=0.4, cuda=False, canvas_size=1280, mag_ratio=1.5, poly=False, show_time=False):

    image = imgproc.loadImage(image_path)

    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size,
                                         mag_ratio, show_time)

    if len(bboxes)==0:
        return []

    bboxes[bboxes < 0] = 0
    min_x_bboxes = np.min(bboxes[:,:,:1],axis=1)
    min_y_bboxes = np.min(bboxes[:,:,1:2],axis=1)
    delt_x_bboxes = np.ceil(np.max(bboxes[:,:,:1],axis=1) - min_x_bboxes)
    delt_y_bboxes = np.ceil(np.max(bboxes[:,:,1:2],axis=1) - min_y_bboxes)

    return np.hstack([min_x_bboxes,min_y_bboxes,delt_x_bboxes, delt_y_bboxes]).astype(int)
    # return np.hstack([bboxes[:, 0], np.ceil(bboxes[:, -2] - bboxes[:, 0])]).astype(int)
