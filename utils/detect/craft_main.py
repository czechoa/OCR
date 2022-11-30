# from datetime import time
import time
import numpy as np
import torch

from utils.detect.craft_pytorch import imgproc
from utils.detect.craft_pytorch.craft import CRAFT
from utils.detect.craft_pytorch.test_net import test_net, copyStateDict


def craft_main(image_path, trained_model='saved_models/craft_mlt_25k.pth', text_threshold=0.7, low_text=0.4,
               link_threshold=0.4, cuda=False, canvas_size=1280, mag_ratio=1.5, poly=False, show_time=False):
    # load net
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + trained_model + ')')

    net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

    net.eval()

    t = time.time()

    image = imgproc.loadImage(image_path)

    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size,
                                         mag_ratio, show_time)

    # score_text_list.append(score_text)
    print("elapsed time : {}s".format(time.time() - t))

    return np.hstack([bboxes[:, 0], np.ceil(bboxes[:, -2] - bboxes[:, 0])]).astype(int)
