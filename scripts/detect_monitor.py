import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scripts.config import *
from scripts.detect_text import read_polys_result, save_polys_result

RESULT_DIR_PATH = os.path.join(BASE_PATH, 'result/')
Path(RESULT_DIR_PATH).mkdir(parents=True, exist_ok=True)
MONITOR_DIR_PATH = os.path.join(RESULT_DIR_PATH, 'monitor/')
Path(MONITOR_DIR_PATH).mkdir(parents=True, exist_ok=True)


def read_img(img_path):
    img = cv2.imread(img_path)
    img_name, img_ext = os.path.splitext(os.path.basename(img_path))
    return img, img_name


def detect_monitor(img, img_name, args):
    poly = get_monitor_gui_result(img, args)
    monitor_img = projective_transformation(img, poly, img_name, args)
    return monitor_img


def projective_transformation(img, poly, img_name, args):
    dst_size = [960,1280]
    dst = []
    pts1 = np.float32(poly)
    pts2 = np.float32([[0,0],[0,dst_size[1]],[dst_size[0],dst_size[1]],[dst_size[0],0]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (dst_size[0],dst_size[1]))
    dst = dst.transpose(1, 0, 2)

    if args.test:
        monitor_img_path = os.path.join(MONITOR_DIR_PATH, img_name) + '_monitor.jpg'
        plt.imsave(monitor_img_path, dst)

    return dst


def create_gui_window(img, param):
    poly = []
    def printCoor(event,x,y,flags,param):
        nonlocal img
        nonlocal img_mes
        nonlocal poly

        if event == cv2.EVENT_LBUTTONDOWN:
            img_tmp = img_mes.copy()

            poly.append([x, y])

            cv2.putText(img_tmp, text=f'(x,y):({x},{y})',org=(x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255,255,255),thickness=1,lineType=cv2.LINE_4)

            print(f'start x:{x}, y:{y}')

            cv2.imshow('image',img_tmp)

        elif event == cv2.EVENT_RBUTTONDOWN:
            param['poly'] = np.array([[p] for p in poly], dtype=np.float32)

    img_mes = img.copy()
    print('Quit -> ESC Key ')

    cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('image',printCoor, param)
    cv2.moveWindow('image',0,0)
    cv2.putText(img_mes, text=f'Quit -> ESC Key',org=(5,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.3, color=(255,255,255),thickness=1,lineType=cv2.LINE_4)

    cv2.imshow('image',img)
    cv2.imshow('image',img_mes)


def get_monitor_gui_result(img, args):
    res_file_path = os.path.join(MONITOR_DIR_PATH, "res_polys") + '.txt'

    if os.path.isfile(res_file_path) and args.detect_monitor_gui:
        poly = read_polys_result(res_file_path)
    else:
        poly = run_monitor_gui(img, res_file_path)

    return poly


def run_monitor_gui(img, res_file_path):
    param = {'poly':[]}
    create_gui_window(img, param)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 27: # when key is Esc
            break

    cv2.destroyAllWindows()

    save_polys_result([param['poly']], res_file_path)

    return param['poly']


def process(img_path, args):
    img, img_name = read_img(img_path)
    monitor_img = detect_monitor(img, img_name, args)

    return monitor_img, img_name
