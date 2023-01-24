import glob
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from craft_pytorch import craft_utils, file_utils, imgproc
from craft_pytorch.craft import CRAFT
from torch.autograd import Variable


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


def inference(monitor_path_list, trained_model_path, craft_dir_path, args):
    """ For test images in a folder """
    # load net
    net = CRAFT() # initialize

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    t = time.time()

    # load data
    result = []

    for k, image_path in enumerate(monitor_path_list):
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args)

        if args.save_txt:
            # save score text
            file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=craft_dir_path)

        result.append(polys)

    return result


def trim_with_craft_result(monitor_path_list, result, craft_dir_path, args):
    craft_imgs = []

    for img_path, polys in zip(monitor_path_list, result):
        img = cv2.imread(img_path)
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))

            temp_left = poly[0]
            temp_right = poly[2]
            temp_top = poly[1]
            temp_bottom = poly[5]

            trimed_img = img[temp_top: temp_bottom, temp_left : temp_right]
            craft_imgs.append(trimed_img)

            if args.save_craft:
                file_name = ''.join(img_path.split('/')[-1].split('.')[-2])
                file_path = os.path.join(craft_dir_path, file_name)
                cv2.imwrite(f'{file_path}_{i}.jpg', trimed_img)

    return craft_imgs


def main(BASE_PATH, result_dir_path, args):
    monitor_dir_path = os.path.join(result_dir_path, 'monitor/')
    monitor_path_list = glob.glob(monitor_dir_path + '*.jpg')

    craft_dir_path = os.path.join(result_dir_path + 'craft/')
    Path(craft_dir_path).mkdir(parents=True, exist_ok=True)
    os.system(f'rm -rf {craft_dir_path}/*')

    trained_model_path = check_craft_model(BASE_PATH)

    result = inference(monitor_path_list, trained_model_path, craft_dir_path, args)
    craft_imgs = trim_with_craft_result(monitor_path_list, result, craft_dir_path, args)

    return craft_imgs
