import glob
import os
import time
from collections import OrderedDict
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import *
from craft_pytorch import craft_utils, file_utils, imgproc
from craft_pytorch.craft import CRAFT
from torch.autograd import Variable

RESULT_DIR_PATH = os.path.join(BASE_PATH, 'result/')
Path(RESULT_DIR_PATH).mkdir(parents=True, exist_ok=True)


def download_and_unzip_craft_model(model_dir, detection_model):
    file_name = detection_model['filename']
    url = detection_model['url']
    zip_path = os.path.join(model_dir, 'temp.zip')

    urlretrieve(url, zip_path)
    with ZipFile(zip_path, 'r') as zipObj:
        zipObj.extract(file_name, model_dir)

    os.remove(zip_path)


def check_craft_model(BASE_PATH):
    model_dir_path = os.path.join(BASE_PATH, 'model/')
    Path(model_dir_path).mkdir(parents=True, exist_ok=True)

    detection_model = {
        'filename': 'craft_mlt_25k.pth',
        'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip',
        'md5sum': '2f8227d2def4037cdb3b34389dcf9ec1'
    }

    detector_path = os.path.join(model_dir_path, detection_model['filename'])

    if os.path.isfile(detector_path) == False:
        download_and_unzip_craft_model(model_dir_path, detection_model)

    return detector_path


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


def test_net(net, image, args):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
                                                        image,
                                                        args.canvas_size,
                                                        interpolation=cv2.INTER_LINEAR,
                                                        mag_ratio=args.mag_ratio
                                                    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if args.cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
                        score_text, score_link,
                        args.text_threshold,
                        args.link_threshold,
                        args.low_text,
                        args.poly
                    )

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

    if args.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


class Inference():
    def __init__(self, trained_model_path, args) -> None:
        """ For test images in a folder """
        self.args = args
        # load net
        self.net = CRAFT() # initialize

        if args.cuda:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model_path)))
        else:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location='cpu')))

        if args.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

    def inference(self, img_path, result_dir_path, i):
        image = imgproc.loadImage(img_path)

        bboxes, polys, score_text = test_net(self.net, image, self.args)

        if self.args.test:
            # save score text
            file_utils.saveResult(img_path, image[:,:,::-1], polys, i, dirname=result_dir_path)

        return polys


def trim_with_craft_result(img_path_list, result, save_dir_path, args):
    craft_imgs_list = []

    for img_path, polys in zip(img_path_list, result):
        img = cv2.imread(img_path)
        craft_imgs = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))

            temp_left = poly[0]
            temp_right = poly[2]
            temp_top = poly[1]
            temp_bottom = poly[5]

            trimed_img = img[temp_top: temp_bottom, temp_left : temp_right]
            craft_imgs.append(trimed_img)

            if args.test:
                file_name, file_ext = os.path.splitext(os.path.basename(img_path))
                file_path = os.path.join(save_dir_path, file_name)
                cv2.imwrite(f'{file_path}_{i:08}.jpg', trimed_img)
        craft_imgs_list.append(craft_imgs)

    return craft_imgs_list


def run(args):
    monitor_dir_path = os.path.join(RESULT_DIR_PATH, 'monitor/')
    monitor_path_list = glob.glob(monitor_dir_path + '*.jpg')

    craft_dir_path = os.path.join(RESULT_DIR_PATH + 'craft/')
    Path(craft_dir_path).mkdir(parents=True, exist_ok=True)
    os.system(f'rm -rf {craft_dir_path}/*')

    trained_model_path = check_craft_model(BASE_PATH)

    inf = Inference(trained_model_path, args)

    polys_list = []
    for i, monitor_path in enumerate(monitor_path_list):
        polys = inf.inference(monitor_path, craft_dir_path, i)
        polys_list.append(polys)

    craft_imgs_list = trim_with_craft_result(monitor_path_list, polys_list, craft_dir_path, args)

    return polys_list, craft_imgs_list
