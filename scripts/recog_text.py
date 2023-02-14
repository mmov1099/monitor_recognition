import copy
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scripts.my_net as mynet
from PIL import Image
from scripts.config import *

RESULT_DIR_PATH = os.path.join(BASE_PATH, 'result/')
Path(RESULT_DIR_PATH).mkdir(parents=True, exist_ok=True)

MODEL_DIR_PATH = os.path.join(BASE_PATH, 'model/')

TABLE_DIR_PATH = os.path.join(RESULT_DIR_PATH, 'table/')
Path(TABLE_DIR_PATH).mkdir(parents=True, exist_ok=True)
os.system(f'rm -rf {TABLE_DIR_PATH}*')


def cv2pil(image):
    new_image = image.copy()

    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)

    new_image = Image.fromarray(new_image)

    return new_image


def save_table(table, img_name):
    index = ['設定(rpm)', '指令(rpm)', '実績(rpm)', \
                '速度(m/s)', 'トルク(%)', 'ループ補正(RPM)']
    columns = ['M1H', 'M2H', 'F1H', 'F2V', 'F3H', \
                'F4V', 'F5H', 'F6V', 'F7H']

    df = pd.DataFrame(index=index, columns = columns, data=table)

    csv_name = img_name + '_table'
    csv_path = os.path.join(TABLE_DIR_PATH, csv_name) + '.csv'

    df.to_csv(csv_path)


class Result2Table():
    def __init__(self, args):
        self.args = args
        self.net = mynet.Inference(args)

        self.gray_dir_path = os.path.join(RESULT_DIR_PATH, 'gray/')
        Path(self.gray_dir_path).mkdir(parents=True, exist_ok=True)
        os.system(f'rm -rf {self.gray_dir_path}*')

    def _get_transcript(self, img):
        transcript = ''

        for j, im in enumerate(img):
            im = im.astype(np.float32)
            result = self.net.inference(im)

            # たまに「.」が認識されないためルールベースで追加
            # 要モデルの改善
            if j == self.args.num_digits-2 and result == '' and not(transcript == ''):
                result = '.'
            transcript += result

        return transcript

    def get_table(self, polys, imgs, img_name):
        imgs_copy = copy.deepcopy(imgs)
        imgs = []
        for start in range(0, len(imgs_copy), self.args.num_digits):
            imgs.append(imgs_copy[start:start+self.args.num_digits])

        table = []
        temp_row = []
        pre_top = -1
        pre_bottom = -1

        for i, (poly, img) in enumerate(zip(polys, imgs)):
            transcript = self._get_transcript(img)

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
        save_table(table, img_name)

        return table
