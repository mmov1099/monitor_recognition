import glob
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from craft_pytorch import craft_utils, file_utils, imgproc
from craft_pytorch.craft import CRAFT
from torch.autograd import Variable

RESULT_DIR_PATH = os.path.join(BASE_PATH, 'result/')
Path(RESULT_DIR_PATH).mkdir(parents=True, exist_ok=True)

GUI_DIR_PATH = os.path.join(RESULT_DIR_PATH, 'gui/')
Path(GUI_DIR_PATH).mkdir(parents=True, exist_ok=True)
os.system(f'rm -rf {GUI_DIR_PATH}/*.jpg')


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

        if self.args.save_txt:
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

            if args.save_craft:
                file_name, file_ext = os.path.splitext(os.path.basename(img_path))
                file_path = os.path.join(save_dir_path, file_name)
                cv2.imwrite(f'{file_path}_{i:08}.jpg', trimed_img)
        craft_imgs_list.append(craft_imgs)

    return craft_imgs_list


def run_craft(args):
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


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()

    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)

    new_image = Image.fromarray(new_image)

    return new_image


def save_table(table, table_dir_path, img_name):
    index = ['設定(rpm)', '指令(rpm)', '実績(rpm)', \
                '速度(m/s)', 'トルク(%)', 'ループ補正(RPM)']
    columns = ['M1H', 'M2H', 'F1H', 'F2V', 'F3H', \
                'F4V', 'F5H', 'F6V', 'F7H']

    df = pd.DataFrame(index=index, columns = columns, data=table)

    csv_name = img_name + '_table'
    csv_path = os.path.join(table_dir_path, csv_name) + '.csv'

    df.to_csv(csv_path)


def result2table(polys_list, imgs_list, img_name_list, args):
    model_dir_path = os.path.join(BASE_PATH, 'model/')
    table_dir_path = os.path.join(RESULT_DIR_PATH, 'table/')
    Path(table_dir_path).mkdir(parents=True, exist_ok=True)

    if args.ocr_type == 'easyocr':
        import easyocr
        reader = easyocr.Reader(['ja', 'en'], model_storage_directory=model_dir_path)
    elif args.ocr_type == 'mangaocr':
        from manga_ocr import MangaOcr
        reader = MangaOcr()
    elif args.ocr_type == 'tesseract':
        import pytesseract
    elif args.ocr_type == 'pyocr':
        import pyocr

        # ツール読み込み
        tools = pyocr.get_available_tools()
        tool = tools[0]

    if args.use_gray:
        gray_dir_path = os.path.join(RESULT_DIR_PATH, 'gray/')
        Path(gray_dir_path).mkdir(parents=True, exist_ok=True)
        os.system(f'rm -rf {RESULT_DIR_PATH}/gray_dir_path/*')

    if args.craft_recog:
        trained_model_path = check_craft_model(BASE_PATH)
        gui_craft_dir_path = os.path.join(RESULT_DIR_PATH, 'gui_craft/')
        os.system(f'rm -rf {gui_craft_dir_path}*')

        inf = Inference(trained_model_path, args)

    table = []
    for polys, imgs, img_name in zip(polys_list, imgs_list, img_name_list):
        pre_top = -1
        pre_bottom = -1
        temp_row = []
        for i, (poly, img) in enumerate(zip(polys, imgs)):
            if args.use_gray:
                gray_img_path = gray_dir_path+img_name+f'_{i}.jpg'
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, img = cv2.threshold(img, args.recog_gray, 255, cv2.THRESH_BINARY)
                img=255-img
                cv2.imwrite(gray_img_path, img)
            if args.craft_recog:
                craft_polys = inf.inference(gray_img_path, gui_craft_dir_path, i)[0]
                craft_polys = np.array(craft_polys).astype(np.int32)
                img = img[craft_polys[0][1]: craft_polys[2][1], craft_polys[0][0]: craft_polys[1][0]]
                cv2.imwrite(gui_craft_dir_path+img_name+f'_{i}.jpg', img)

            if args.ocr_type == 'easyocr':
                transcript = reader.readtext(img, detail=0, text_threshold=0.3)
                if len(transcript)==1:
                    transcript = transcript[0]
                elif len(transcript)==0:
                    transcript=''
            elif args.ocr_type == 'mangaocr':
                pil_img = cv2pil(img)
                transcript = reader(pil_img)
            elif args.ocr_type == 'tesseract':
                pil_img = cv2pil(img)
                transcript = pytesseract.image_to_string(pil_img, lang='eng',
                                config='--psm 7 --oem 1 -c tessedit_char_whitelist="0123456789."').strip()
                transcript = re.sub(r'\D', '', transcript)
            elif args.ocr_type == 'pyocr':
                pil_img = cv2pil(img)
                # OCR
                builder = pyocr.builders.DigitBuilder(tesseract_layout=8)
                transcript = tool.image_to_string(pil_img, lang='eng', builder=builder)
                # 数値以外の文字を除去
                transcript = re.sub(r'\D', '', transcript)

            temp_top = poly[0][1]
            temp_bottom = poly[2][1]

            if len(temp_row)==0:
                temp_row = [transcript]
            elif (pre_top-10 <= temp_top and temp_top <= pre_top+10) and \
                    (pre_bottom-10 <= temp_bottom and temp_bottom <= pre_bottom+10):
                temp_row.append(transcript)
            else:
                table.append(temp_row)
                temp_row = [transcript]

            pre_top = temp_top
            pre_bottom = temp_bottom
        table.append(temp_row)
        print(table)
        save_table(table, table_dir_path, img_name)

    return table


def create_gui_window(img, img_name, param, height, width, args):
    idx = 0

    def printCoor(event,x,y,flags,param):
        # OpenCVマウスイベントのcallbackは上記のような引数をとる。

        #　nonlocalの宣言をして、この関数外にある変数にアクセス。
        nonlocal img
        nonlocal img_mes
        nonlocal img_name
        nonlocal idx

        if event == cv2.EVENT_LBUTTONDOWN:
            # 元の画像に直接書き込むと前の描画がそのまま残ってしまうため、コピーを作成。
            img_tmp = img_mes.copy()

            # 直線で書きたい場合
            temp_y = y
            for _ in range(args.row):
                temp_x = x
                for _ in range(args.column):
                    cv2.line(img_tmp,(temp_x,temp_y),(temp_x+width,temp_y),(255,255,255),1)
                    cv2.line(img_tmp,(temp_x,temp_y),(temp_x,temp_y+height),(255,255,255),1)
                    temp_x += width
                    temp_y += args.tilt
                temp_y += (height-args.tilt*args.column)

            # cv2.rectangle(img_tmp,(x,y),(x+width,y+height),(255,255,255), thickness=1)

            # 座標は左上が原点　x座標:左から右　y座標：上から下　行列では,行(height):y、列(width):x
            # orgは文字オブジェクトの左下の座標
            cv2.putText(img_tmp, text=f'(x,y):({x},{y})',org=(x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255,255,255),thickness=1,lineType=cv2.LINE_4)

            print(f'start x:{x}, y:{y} --height:{height},width:{width}-- end x:{x+width}, y:{y+height},{idx}')

            cv2.imshow('image',img_tmp)

        elif event == cv2.EVENT_RBUTTONDOWN:
            temp_y = y
            for _ in range(args.row):
                temp_x = x
                for _ in range(args.column):
                    trim_name = os.path.join(GUI_DIR_PATH, f'{img_name}_trim_{idx:08}.jpg')
                    trim_array = trim(array=img, x=temp_x, y=temp_y, width=width, height=height)
                    param['polys'].append(np.array([[temp_x, temp_y],
                                                    [temp_x+width, temp_y],
                                                    [temp_x+width, temp_y+height],
                                                    [temp_x, temp_y+height]], dtype=np.float32))
                    param['clipped_imgs'].append(trim_array)
                    cv2.imwrite(trim_name, trim_array)

                    temp_x += width
                    idx += 1
                temp_y += (height+args.tilt)

    def printCoor_each(event,x,y,flags,param):
        # OpenCVマウスイベントのcallbackは上記のような引数をとる。

        #　nonlocalの宣言をして、この関数外にある変数にアクセス。
        nonlocal img
        nonlocal img_mes
        nonlocal img_name
        nonlocal idx

        if event == cv2.EVENT_LBUTTONDOWN:
            # 元の画像に直接書き込むと前の描画がそのまま残ってしまうため、コピーを作成。
            img_tmp = img_mes.copy()

            # 直線で書きたい場合
            cv2.line(img_tmp,(x,y),(x+width,y),(255,255,255),1)
            cv2.line(img_tmp,(x,y),(x,y+height),(255,255,255),1)
            cv2.line(img_tmp,(x+width,y),(x+width,y+height),(255,255,255),1)
            cv2.line(img_tmp,(x,y+height),(x+width,y+height),(255,255,255),1)

            # cv2.rectangle(img_tmp,(x,y),(x+width,y+height),(255,255,255), thickness=1)

            # 座標は左上が原点　x座標:左から右　y座標：上から下　行列では,行(height):y、列(width):x
            # orgは文字オブジェクトの左下の座標
            cv2.putText(img_tmp, text=f'(x,y):({x},{y})',org=(x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255,255,255),thickness=1,lineType=cv2.LINE_4)

            print(f'start x:{x}, y:{y} --height:{height},width:{width}-- end x:{x+width}, y:{y+height},{idx}')

            cv2.imshow('image',img_tmp)

        elif event == cv2.EVENT_RBUTTONDOWN:
            trim_name = os.path.join(GUI_DIR_PATH, f'{img_name}_trim_{idx:08}.jpg')
            trim_array = trim(array=img, x=x, y=y, width=width, height=height)
            param['polys'].append(np.array([[x, y], [x+width, y], [x+width, y+height], [x, y+height]], dtype=np.float32))
            param['clipped_imgs'].append(trim_array)
            cv2.imwrite(trim_name, trim_array)
            idx += 1

    img_mes = img.copy()
    print('Quit -> ESC Key ')

    cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
    if args.text_detect == 'gui':
        cv2.setMouseCallback('image',printCoor, param)
    elif args.text_detect == 'gui_each':
        cv2.setMouseCallback('image',printCoor_each, param)
    cv2.moveWindow('image',100,100) #100,100はwindows上に表示される位置を指定。
    cv2.putText(img_mes, text=f'Quit -> ESC Key',org=(5,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.3, color=(255,255,255),thickness=1,lineType=cv2.LINE_4)

    cv2.imshow('image',img)
    cv2.imshow('image',img_mes)
    # 第一引数の名前が同じだと同じウィンドウに上書き表示(名前が異なると別のウインドウが作成される)。


def get_gui_result(img, img_name, args):
    res_file_path = os.path.join(GUI_DIR_PATH, "res_polys") + '.txt'

    if os.path.isfile(res_file_path) and args.run_gui:
        polys = read_polys_result(res_file_path)

        clipped_imgs = []
        for idx, poly in enumerate(polys):
            trim_name = os.path.join(GUI_DIR_PATH, f'{img_name}_trim_{idx:08}.jpg')

            x = int(poly[0][0])
            y = int(poly[0][1])

            trim_array = trim(array=img, x=x, y=y, width=args.width, height=args.height)
            clipped_imgs.append(trim_array)
            cv2.imwrite(trim_name, trim_array)

    else:
        polys, clipped_imgs = run_gui(img, img_name, res_file_path, args)

    return polys, clipped_imgs


def run_gui(img, img_name, res_file_path, args):
    height = args.height
    width = args.width
    param = {'polys':[], 'clipped_imgs':[]}
    create_gui_window(img, img_name, param, height, width, args)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 27: # when key is Esc
            break
        elif key == ord('h'):
            cv2.destroyAllWindows()
            print(f'height is {height} now')
            height = int(input('input new height value\n>>>'))
            param = {'polys':[], 'clipped_imgs':[]}
            create_gui_window(img, img_name, param, height, width, args)
        elif key == ord('w'):
            cv2.destroyAllWindows()
            print(f'width is {width} now')
            width = int(input('input new width value\n>>>'))
            param = {'polys':[], 'clipped_imgs':[]}
            create_gui_window(img, img_name, param, height, width, args)

    cv2.destroyAllWindows()

    save_polys_result(param['polys'], res_file_path)

    return param['polys'], param['clipped_imgs']


def trim(array, x, y, width, height):
    array_trim = array.copy()
    array_trim = array_trim[y:y + height, x:x+width]

    return array_trim


def save_polys_result(boxes, res_file_path):
    with open(res_file_path, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            f.write(strResult)


def read_polys_result(res_file_path):
    polys = []

    with open(res_file_path, 'r') as f:
        for box in f.read().splitlines():
            box = box.split(',')
            poly = np.array([int(b) for b in box], dtype=np.float32).reshape((4, 2))
            polys.append(poly)

    return polys
