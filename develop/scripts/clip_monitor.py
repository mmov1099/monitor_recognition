import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scripts.config import *

RESULT_DIR_PATH = os.path.join(BASE_PATH, 'result/')
Path(RESULT_DIR_PATH).mkdir(parents=True, exist_ok=True)
CONTOURS_DIR_PATH = os.path.join(RESULT_DIR_PATH, 'contours/')
Path(CONTOURS_DIR_PATH).mkdir(parents=True, exist_ok=True)
MONITOR_DIR_PATH = os.path.join(RESULT_DIR_PATH, 'monitor/')
Path(MONITOR_DIR_PATH).mkdir(parents=True, exist_ok=True)

from detect_text import read_polys_result, save_polys_result


def read_img(img_path, args):
    # 8ビット1チャンネルのグレースケールとして画像を読み込む
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray_img = cv2.threshold(gray, args.monitor_gray, 255, cv2.THRESH_BINARY)
    img_name, img_ext = os.path.splitext(os.path.basename(img_path))

    return img, gray_img, img_name


def detect_monitor(img, gray_img, img_name, args):
    if args.detect_monitor_type == 'find_contours':
        #輪郭検出
        contours, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 面積で選別
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if args.min_area < area and area < args.max_area:
                epsilon = 0.1*cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                areas.append(approx)
        poly = areas[0]

        if args.test:
            contours_img = cv2.drawContours(img, areas, -1,(255,0,0),3)
            contours_img_path = os.path.join(CONTOURS_DIR_PATH, img_name) + '_contours.jpg'
            plt.imsave(contours_img_path, contours_img)

    elif args.detect_monitor_type == 'gui':
        poly = get_monitor_gui_result(img, args)

    monitor_img = projective_transformation(img, poly, img_name, args)

    return monitor_img


def projective_transformation(img, poly, img_name, args):
    # 射影変換
    dst_size = [960,1280]   # 射影変換後の画像サイズ
    dst = []
    pts1 = np.float32(poly)   # 抽出した領域の四隅の座標
    pts2 = np.float32([[0,0],[0,dst_size[1]],[dst_size[0],dst_size[1]],[dst_size[0],0]])   # 射影変換後の四隅の座標

    # ホモグラフィ行列を求め、射影変換する
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (dst_size[0],dst_size[1]))
    dst = dst.transpose(1, 0, 2)

    if args.test:
        monitor_img_path = os.path.join(MONITOR_DIR_PATH, img_name) + '_monitor.jpg'
        plt.imsave(monitor_img_path, dst)

    return dst


def create_gui_window(img, param, args):
    poly = []
    def printCoor(event,x,y,flags,param):
        # OpenCVマウスイベントのcallbackは上記のような引数をとる。

        #　nonlocalの宣言をして、この関数外にある変数にアクセス。
        nonlocal img
        nonlocal img_mes
        nonlocal poly

        if event == cv2.EVENT_LBUTTONDOWN:
            # 元の画像に直接書き込むと前の描画がそのまま残ってしまうため、コピーを作成。
            img_tmp = img_mes.copy()

            poly.append([x, y])
            # 座標は左上が原点　x座標:左から右　y座標：上から下　行列では,行(height):y、列(width):x
            # orgは文字オブジェクトの左下の座標
            cv2.putText(img_tmp, text=f'(x,y):({x},{y})',org=(x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255,255,255),thickness=1,lineType=cv2.LINE_4)

            print(f'start x:{x}, y:{y} \n {poly}')

            cv2.imshow('image',img_tmp)

        elif event == cv2.EVENT_RBUTTONDOWN:
            param['poly'] = np.array([[p] for p in poly], dtype=np.float32)

    img_mes = img.copy()
    print('Quit -> ESC Key ')

    cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('image',printCoor, param)
    cv2.moveWindow('image',100,100) #100,100はwindows上に表示される位置を指定。
    cv2.putText(img_mes, text=f'Quit -> ESC Key',org=(5,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.3, color=(255,255,255),thickness=1,lineType=cv2.LINE_4)

    cv2.imshow('image',img)
    cv2.imshow('image',img_mes)
    # 第一引数の名前が同じだと同じウィンドウに上書き表示(名前が異なると別のウインドウが作成される)。


def get_monitor_gui_result(img, args):
    res_file_path = os.path.join(MONITOR_DIR_PATH, "res_polys") + '.txt'

    if os.path.isfile(res_file_path) and args.detect_monitor_gui:
        poly = read_polys_result(res_file_path)
    else:
        poly = run_monitor_gui(img, res_file_path, args)

    return poly


def run_monitor_gui(img, res_file_path, args):
    param = {'poly':[]}
    create_gui_window(img, param, args)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 27: # when key is Esc
            break
        elif key == ord('x'):
            param['poly'].pop(-1)
            create_gui_window(img, param, args)

    cv2.destroyAllWindows()

    save_polys_result([param['poly']], res_file_path)

    return param['poly']


def process(img_path, args):
    img, gray_img, img_name = read_img(img_path, args)
    monitor_img = detect_monitor(img, gray_img, img_name, args)

    return monitor_img, img_name
