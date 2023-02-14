import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyocr
from google.cloud import vision
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import my_net
import pytesseract
from config import *
from craft import Inference, check_craft_model

RESULT_DIR_PATH = os.path.join(BASE_PATH, 'result/')
Path(RESULT_DIR_PATH).mkdir(parents=True, exist_ok=True)

MODEL_DIR_PATH = os.path.join(BASE_PATH, 'model/')

TABLE_DIR_PATH = os.path.join(RESULT_DIR_PATH, 'table/')
Path(TABLE_DIR_PATH).mkdir(parents=True, exist_ok=True)
os.system(f'rm -rf {TABLE_DIR_PATH}*')


def cv2pil(image):
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


class Result2Table():
    def __init__(self, args):
        self.args = args
        if args.ocr_type == 'easyocr':
            import easyocr
            self.reader = easyocr.Reader(['ja', 'en'], model_storage_directory=MODEL_DIR_PATH)
        elif args.ocr_type == 'mangaocr':
            from manga_ocr import MangaOcr
            self.reader = MangaOcr()
        elif args.ocr_type == 'pyocr':
            tools = pyocr.get_available_tools()
            self.tool = tools[0]
        elif args.ocr_type == 'gcv':
            self.client = vision.ImageAnnotatorClient.from_service_account_json(SERVICE_ACCOUNT_FILE)
        elif args.ocr_type == 'mynet':
            self.net = my_net.Inference(args.mynet_type)

        if args.use_gray:
            self.gray_dir_path = os.path.join(RESULT_DIR_PATH, 'gray/')
            Path(self.gray_dir_path).mkdir(parents=True, exist_ok=True)
            # os.system(f'rm -rf {gray_dir_path}*')

        if args.craft_recog:
            trained_model_path = check_craft_model(BASE_PATH)
            self.gui_craft_dir_path = os.path.join(RESULT_DIR_PATH, 'gui_craft/')
            Path(self.gui_craft_dir_path).mkdir(parents=True, exist_ok=True)
            os.system(f'rm -rf {self.gui_craft_dir_path}*')

            self.inf = Inference(trained_model_path, args)

    def _get_transcript(self, img, img_name, i):
        transcript = ''

        if self.args.ocr_type == 'easyocr':
            transcript = self.reader.readtext(img, detail=0, text_threshold=0.3)
            if len(transcript)==1:
                transcript = transcript[0]
            elif len(transcript)==0:
                transcript=''
        elif self.args.ocr_type == 'mangaocr':
            pil_img = cv2pil(img)
            transcript = self.reader(pil_img)
        elif self.args.ocr_type == 'tesseract':
            transcript = ''
            pil_img = cv2pil(img)
            transcript = pytesseract.image_to_string(pil_img, lang='eng',
                            config='--psm 6 --oem 3 -c tessedit_char_whitelist="0123456789.-"').strip()
        elif self.args.ocr_type == 'pyocr':
            pil_img = cv2pil(img)
            # OCR
            builder = pyocr.builders.DigitBuilder(tesseract_layout=8)
            transcript = self.tool.image_to_string(pil_img, lang='eng', builder=builder)
            # 数値以外の文字を除去
            transcript = re.sub(r'\D', '', transcript)
        elif self.args.ocr_type == 'gcv':
            is_success, im_buf_arr = cv2.imencode(".jpg", img)
            byte_im = im_buf_arr.tobytes()
            image = vision.Image(content=byte_im)
            response = self.client.text_detection(
                image=image,
                image_context={"language_hints": ["en"]},
            )
            transcript = response.text_annotations[1:]
        elif self.args.ocr_type == 'mynet':
            for j, im in enumerate(img):
                gray_img_path = self.gray_dir_path+img_name+f'_{i}_{j}.jpg'
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                ret, im = cv2.threshold(im, self.args.recog_gray, 255, cv2.THRESH_BINARY)
                im=255-im
                if self.args.test:
                    cv2.imwrite(gray_img_path, im)
                im = im.astype(np.float32)
                result = self.net.inference(im)
                if j == self.args.num_digits-2 and result == '' and not(transcript == ''):
                    result = '.'
                transcript += result

        return transcript

    def get_table(self, polys, imgs, img_name):
        if self.args.ocr_type == 'mynet':
            import copy
            imgs_copy = copy.deepcopy(imgs)
            imgs = []
            for start in range(0, len(imgs_copy), self.args.num_digits):
                imgs.append(imgs_copy[start:start+self.args.num_digits])

        table = []
        temp_row = []
        pre_top = -1
        pre_bottom = -1

        for i, (poly, img) in enumerate(zip(polys, imgs)):
            if self.args.use_gray and not(self.args.ocr_type == 'mynet'):
                gray_img_path = self.gray_dir_path+img_name+f'_{i}.jpg'
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, img = cv2.threshold(img, self.args.recog_gray, 255, cv2.THRESH_BINARY)
                img=255-img
                if self.args.test:
                    cv2.imwrite(gray_img_path, img)

            if self.args.craft_recog:
                craft_polys = self.inf.inference(gray_img_path, self.gui_craft_dir_path, i)[0]
                craft_polys = np.array(craft_polys).astype(np.int32)
                img = img[craft_polys[0][1]: craft_polys[2][1], craft_polys[0][0]: craft_polys[1][0]]
                if self.args.test:
                    cv2.imwrite(self.gui_craft_dir_path+img_name+f'_{i}.jpg', img)

            if self.args.super_resolution != 1:
                img = cv2.resize(img, (img.shape[1]*self.args.super_resolution, img.shape[0]*self.args.super_resolution), interpolation=cv2.INTER_CUBIC)
                RESIZED_DIR_PATH = os.path.join(RESULT_DIR_PATH, 'resized/')
                Path(RESIZED_DIR_PATH).mkdir(parents=True, exist_ok=True)
                if self.args.test:
                    cv2.imwrite(RESIZED_DIR_PATH+img_name+f'_{i}.jpg', img)

            transcript = self._get_transctipt(img, img_name, i)

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
        save_table(table, TABLE_DIR_PATH, img_name)

        return table
