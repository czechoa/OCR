"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from utils.detect.craft_pytorch import craft_utils
from utils.detect.craft_pytorch import imgproc
from utils.detect.craft_pytorch.craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio, show_time,
             refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


# def main(image_path, trained_model='weights/craft_mlt_25k.pth', text_threshold=0.7, low_text=0.4, link_threshold=0.4,
#          cuda=False, canvas_size=1280, mag_ratio=1.5, poly=False, show_time=False, test_folder='/data/', refine=False,
#          refiner_model='weights/craft_refiner_CTW1500.pth'):
#     # load net
#     net = CRAFT()  # initialize
#
#     print('Loading weights from checkpoint (' + trained_model + ')')
#     if cuda:
#         net.load_state_dict(copyStateDict(torch.load(trained_model)))
#     else:
#         net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))
#
#     if cuda:
#         net = net.cuda()
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = False
#
#     net.eval()
#
#     # LinkRefiner
#     refine_net = None
#     if refine:
#         from refinenet import RefineNet
#         refine_net = RefineNet()
#         print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
#         if cuda:
#             refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
#             refine_net = refine_net.cuda()
#             refine_net = torch.nn.DataParallel(refine_net)
#         else:
#             refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))
#
#         refine_net.eval()
#         poly = True
#
#     t = time.time()
#
#     # load data
#     score_text_list = []
#
#     # for k, image_path in enumerate(image_list):
#     image = imgproc.loadImage(image_path)
#
#
#
#     bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly,
#                                          canvas_size, mag_ratio, show_time, refine_net)
#
#         # score_text_list.append(score_text)
#     print("elapsed time : {}s".format(time.time() - t))
#
#
#     return np.hstack([bboxes[:, 0], np.ceil(bboxes[:, -2] - bboxes[:, 0])]).astype(int)
#         # save score text
#         # filename, file_ext = os.path.splitext(os.path.basename(image_path))
#         # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
#         # cv2.imwrite(mask_file, score_text)
#         #
#         # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
#
#     #
#     # return score_text_list, polys, bboxes
