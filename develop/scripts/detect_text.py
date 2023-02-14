import os
from pathlib import Path

import cv2
import numpy as np
from scripts.config import *

RESULT_DIR_PATH = os.path.join(BASE_PATH, 'result/')
Path(RESULT_DIR_PATH).mkdir(parents=True, exist_ok=True)

GUI_DIR_PATH = os.path.join(RESULT_DIR_PATH, 'gui/')
Path(GUI_DIR_PATH).mkdir(parents=True, exist_ok=True)
os.system(f'rm -rf {GUI_DIR_PATH}/*.jpg')


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
                    if args.test:
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
            for i in range(args.num_digits):
                cv2.line(img_tmp,(x+width//args.num_digits*i,y),(x+width//args.num_digits*i,y+height),(255,255,255),1)

            # cv2.rectangle(img_tmp,(x,y),(x+width,y+height),(255,255,255), thickness=1)

            # 座標は左上が原点　x座標:左から右　y座標：上から下　行列では,行(height):y、列(width):x
            # orgは文字オブジェクトの左下の座標
            cv2.putText(img_tmp, text=f'(x,y):({x},{y})',org=(x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255,255,255),thickness=1,lineType=cv2.LINE_4)

            print(f'start x:{x}, y:{y} --height:{height},width:{width}-- end x:{x+width}, y:{y+height},{idx}')

            cv2.imshow('image',img_tmp)
            # --height:35,width:70--
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i in range(args.num_digits):
                trim_name = os.path.join(GUI_DIR_PATH, f'{img_name}_trim_{idx:08}_{i}.jpg')
                trim_array = trim(array=img, x=x+width//args.num_digits*i, y=y, width=width//args.num_digits, height=height)
                cv2.imwrite(trim_name, trim_array)
                param['clipped_imgs'].append(trim_array)
            param['polys'].append(np.array([[x, y],
                                [x+width, y],
                                [x+width, y+height],
                                [x, y+height]], dtype=np.float32))
            idx += 1

    img_mes = img.copy()
    print('Quit -> ESC Key ')

    cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
    if args.detect_text_type == 'gui':
        cv2.setMouseCallback('image',printCoor, param)
    elif args.detect_text_type == 'gui_each':
        cv2.setMouseCallback('image',printCoor_each, param)
    cv2.moveWindow('image',100,100) #100,100はwindows上に表示される位置を指定。
    cv2.putText(img_mes, text=f'Quit -> ESC Key',org=(5,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.3, color=(255,255,255),thickness=1,lineType=cv2.LINE_4)

    cv2.imshow('image',img)
    cv2.imshow('image',img_mes)
    # 第一引数の名前が同じだと同じウィンドウに上書き表示(名前が異なると別のウインドウが作成される)。


def get_gui_result(img, img_name, args):
    res_file_path = os.path.join(GUI_DIR_PATH, "res_polys") + '.txt'

    if os.path.isfile(res_file_path) and args.detect_text_gui:
        polys = read_polys_result(res_file_path)

        clipped_imgs = []
        for idx, poly in enumerate(polys):
            trim_name = os.path.join(GUI_DIR_PATH, f'{img_name}_trim_{idx:08}.jpg')

            x = int(poly[0][0])
            y = int(poly[0][1])
            width = int(poly[1][0]) - int(poly[0][0])
            height = int(poly[2][1]) - int(poly[0][1])

            if args.detect_text_type == 'gui_each':
                width = width//args.num_digits
                for _ in range(args.num_digits):
                    trim_array = trim(array=img, x=x, y=y, width=width, height=height)
                    clipped_imgs.append(trim_array)
                    if args.test:
                        cv2.imwrite(trim_name, trim_array)
                    x += width

            else:
                trim_array = trim(array=img, x=x, y=y, width=width, height=height)
                clipped_imgs.append(trim_array)
                if args.test:
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
