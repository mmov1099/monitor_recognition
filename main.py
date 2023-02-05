import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter('ignore')

from scripts.config import *

sys.path.append(os.path.join(BASE_PATH, 'scripts/craft_pytorch'))
from scripts import clip_monitor, text_detect


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--img_dir',
        default='imgs/',
        type=str,
        help='folder path to input images'
    )

    # for monitor detection
    parser.add_argument(
        '--monitor_gray',
        default=100,
        type=int,
        help='gray threthold for monitor image processing'
    )
    parser.add_argument(
        '--min_area',
        default=50000,
        type=int,
        help='min area threthold for monitor image processing'
    )
    parser.add_argument(
        '--max_area',
        default=1000000,
        type=int,
        help='max area threthold for monitor image processing'
    )
    parser.add_argument(
        '--save_contours',
        action='store_true',
        help='save contours of monitor in result dir'
    )
    parser.add_argument(
        '--save_monitor',
        action='store_false',
        help='save clipped monitor img in result dir'
    )

    # for text detection
    parser.add_argument(
        '--text_detect',
        default='gui',
        choices=['craft', 'gui', 'gui_each'],
        help='choose text detection type, craft is not updated'
    )
    parser.add_argument(
        '--run_gui',
        action='store_false',
        help='run detection monitor gui forcely'
    )

    # for CRAFT
    parser.add_argument(
        '--model_dir',
        default='model',
        type=str,
        help='CRAFT model dir'
    )
    parser.add_argument(
        '--text_threshold',
        default=-1,
        type=float,
        help='CRAFT text confidence threshold'
    )
    parser.add_argument(
        '--low_text',
        default=0.23,
        type=float,
        help='CRAFT text low-bound score'
    )
    parser.add_argument(
        '--link_threshold',
        default=-1,
        type=float,
        help='CRAFT link confidence threshold'
    )
    parser.add_argument(
        '--cuda',
        action = 'store_false',
        help='Use cuda for inference for CRAFT'
    )
    parser.add_argument(
        '--canvas_size',
        default=1280, type=int,
        help='CRAFT image size for inference'
    )
    parser.add_argument(
        '--mag_ratio',
        default=1.5,
        type=float,
        help='image magnification ratio'
    )
    parser.add_argument(
        '--poly',
        action='store_true',
        help='enable polygon type'
    )
    parser.add_argument(
        '--show_time',
        action='store_true',
        help='show processing time'
    )
    parser.add_argument(
        '--save_txt',
        action='store_true',
        help='save craft result as txt file'
    )
    parser.add_argument(
        '--save_craft',
        action='store_true',
        help='save craft result as jpg file'
    )

    # for text detect gui
    parser.add_argument(
        '--height',
        default=46,
        help='default height value of each cell of monitor table'
    )
    parser.add_argument(
        '--width',
        default=107,
        help='default width value of each cell of monitor table'
    )
    parser.add_argument(
        '--row',
        default=6,
        type=int,
        help='default row number of monitor table'
    )
    parser.add_argument(
        '--column',
        default=9,
        type=int,
        help='default column number of monitor table'
    )
    parser.add_argument(
        '--tilt',
        default=1,
        type=int,
        help='default horizon tilt value of monitor table of cells'
    )

    # for text recognition
    parser.add_argument(
        '--ocr_type',
        default='tesseract',
        choices=['easyocr', 'mangaocr', 'tesseract', 'pyocr'],
        help='choose ocr library'
    )
    parser.add_argument(
        '--use_gray',
        action='store_false',
        help='convert grayscale image from trimed image for text recognition'
    )
    parser.add_argument(
        '--recog_gray',
        default=220,
        type=int,
        help='gray threthold for trimed image'
    )
    parser.add_argument(
        '--craft_recog',
        action='store_false',
        help='use craft after trimming image for text recognition'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = args_parse()

    img_dir_path = os.path.join(BASE_PATH, args.img_dir)

    img_path_list = glob.glob(img_dir_path + '*.jpg')
    img_path_list += glob.glob(img_dir_path + '*.png')

    polys_list = []
    clipped_imgs_list = []
    img_name_list = []
    for img_path in img_path_list:
        monitor_img, img_name = clip_monitor.process(img_path, args)
        img_name_list.append(img_name)
        if args.text_detect == 'gui' or args.text_detect == 'gui_each':
            polys, clipped_imgs = text_detect.get_gui_result(monitor_img, img_name, args)
            polys_list.append(polys)
            clipped_imgs_list.append(clipped_imgs)
    if args.text_detect == 'craft':
        polys_list, clipped_imgs_list = text_detect.run_craft(args)

    monitor_table = text_detect.result2table(polys_list, clipped_imgs_list, img_name_list, args)
