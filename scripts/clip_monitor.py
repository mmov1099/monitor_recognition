import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_img(img_path_list, args):
    img_list = []
    gray_img_list = []
    img_name_list = []

    for img_path in img_path_list:
        # 8ビット1チャンネルのグレースケールとして画像を読み込む
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray_img = cv2.threshold(gray, args.monitor_gray, 255, cv2.THRESH_BINARY)

        img_list.append(img)
        gray_img_list.append(gray_img)
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        img_name_list.append(img_name)

    return img_list, gray_img_list, img_name_list


def detect_monitor(img_list, gray_img_list, img_name_list, result_dir_path, args):
    if args.save_contours:
        contours_dir_path = os.path.join(result_dir_path, 'contours/')
        Path(contours_dir_path).mkdir(parents=True, exist_ok=True)
    else:
        contours_dir_path = None
    if args.save_monitor:
        monitor_dir_path = os.path.join(result_dir_path, 'monitor/')
        Path(monitor_dir_path).mkdir(parents=True, exist_ok=True)
    else:
        monitor_dir_path = None

    monitor_img_list = []

    for img, gray_img, img_name in zip(img_list, gray_img_list, img_name_list):
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

        monitor_img_list.append(projective_transformation(img, areas, img_name, monitor_dir_path, args))

        if args.save_contours:
            contours_img = cv2.drawContours(img, areas, -1,(255,0,0),3)
            contours_img_path = os.path.join(contours_dir_path, img_name) + '_contours.jpg'
            plt.imsave(contours_img_path, contours_img)

    return monitor_img_list


def projective_transformation(img, areas, img_name, monitor_dir_path, args):
    # 射影変換
    dst_size = [960,1280]   # 射影変換後の画像サイズ
    dst = []
    pts1 = np.float32(areas[0])   # 抽出した領域の四隅の座標
    pts2 = np.float32([[0,0],[0,dst_size[1]],[dst_size[0],dst_size[1]],[dst_size[0],0]])   # 射影変換後の四隅の座標

    # ホモグラフィ行列を求め、射影変換する
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (dst_size[0],dst_size[1]))
    dst = dst.transpose(1, 0, 2)

    # img[top : bottom, left : right]
    dst = dst[len(dst)//2-25 : len(dst)-100]

    if args.save_monitor:
        monitor_img_path = os.path.join(monitor_dir_path, img_name) + '_monitor.jpg'
        plt.imsave(monitor_img_path, dst)

    return dst


def process(img_dir_path, result_dir_path, args):
    img_path_list = glob.glob(img_dir_path + '*.jpg')
    img_path_list += glob.glob(img_dir_path + '*.png')

    img_list, gray_img_list, img_name_list = read_img(img_path_list, args)
    monitor_img_list = detect_monitor(img_list, gray_img_list, img_name_list, result_dir_path, args)

    return monitor_img_list, img_name_list
