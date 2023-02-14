import argparse
import glob
import os
import sys
import warnings
from scripts import detect_text, recog_text

warnings.simplefilter('ignore')

from scripts.config import *

sys.path.append(os.path.join(BASE_PATH, 'scripts/craft_pytorch'))
from scripts import clip_monitor, craft


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--img_dir',
        default='imgs/',
        type=str,
        help='folder path to input images'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='save result in each process'
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
        '--detect_monitor_type',
        default='gui',
        choices=['gui', 'find_contours'],
        help='choose monitor detection type'
    )
    parser.add_argument(
        '--detect_monitor_gui',
        action='store_false',
        help='run detect monitor gui forcely'
    )

    # for text detection
    parser.add_argument(
        '--detect_text_type',
        default='gui',
        choices=['craft', 'gui', 'gui_each'],
        help='choose text detection type, craft is not updated'
    )
    parser.add_argument(
        '--detect_text_gui',
        action='store_false',
        help='run detect text gui forcely'
    )
    parser.add_argument(
        '--num_digits',
        default=5,
        type=int,
        help='a number of digits of each cell of monitor'
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
        default=0.15,
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
        default=1.1,
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

    # for text detect gui
    parser.add_argument(
        '--height',
        default=35,
        type=int,
        help='default height value of each cell of monitor table'
    )
    parser.add_argument(
        '--width',
        default=17*5,
        type=int,
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
        default=0,
        type=int,
        help='default horizon tilt value of monitor table of cells'
    )

    # for text recognition
    parser.add_argument(
        '--ocr_type',
        default='tesseract',
        choices=['easyocr', 'mangaocr', 'tesseract', 'pyocr', 'gcv', 'mynet'],
        help='choose ocr library'
    )
    parser.add_argument(
        '--mynet_type',
        default='cnn',
        choices=['cnn', 'mlp'],
        help='choose mynet type'
    )
    parser.add_argument(
        '--use_gray',
        action='store_false',
        help='convert grayscale image from trimed image for text recognition'
    )
    parser.add_argument(
        '--recog_gray',
        default=215,
        type=int,
        help='gray threthold for trimed image'
    )
    parser.add_argument(
        '--craft_recog',
        action='store_false',
        help='use craft after trimming image for text recognition'
    )
    parser.add_argument(
        '--super_resolution',
        default=1,
        type=int,
        help='magnification ratio of super resolution when text recognition'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = args_parse()

    img_dir_path = os.path.join(BASE_PATH, args.img_dir)

    img_path_list = glob.glob(img_dir_path + '*.jpg')
    img_path_list += glob.glob(img_dir_path + '*.png')

    result2table = recog_text.Result2Table(args)
    polys_list = []
    clipped_imgs_list = []
    img_name_list = []
    table_list = []

    if args.detect_text_type == 'gui' or args.detect_text_type == 'gui_each':
        for img_path in img_path_list:
            monitor_img, img_name = clip_monitor.process(img_path, args)

            polys, clipped_imgs = detect_text.get_gui_result(monitor_img, img_name, args)

            table = result2table(polys, clipped_imgs, img_name)
            table_list.append(table)

    elif args.detect_text_type == 'craft': # not updated
        polys_list, clipped_imgs_list = craft.run(args)
