import argparse
import glob
import os
import sys
import warnings

from tqdm import tqdm

warnings.simplefilter('ignore')
from scripts import detect_monitor, detect_text, recog_text
from scripts.config import *

sys.path.append(os.path.join(BASE_PATH, 'scripts/craft_pytorch'))


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
    parser.add_argument(
        '--make_data',
        action='store_true',
        help='make digits data for training mynet'
    )

    # for monitor detection
    parser.add_argument(
        '--detect_monitor_gui',
        action='store_false',
        help='run detect monitor gui forcely'
    )

    # for text detection
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

    # for text detect gui
    parser.add_argument(
        '--height',
        default=35,
        type=int,
        help='default height value of each text of monitor table'
    )
    parser.add_argument(
        '--width',
        default=17,
        type=int,
        help='default width value of each text of monitor table'
    )

    # for recog text
    parser.add_argument(
        '--model_name',
        default='mynet',
        type=str,
        help='the name of a saving text recog model'
    )
    parser.add_argument(
        '--model_type',
        default='cnn',
        choices=['cnn', 'mlp'],
        help='choose mynet type'
    )
    parser.add_argument(
        '--gray_threshold',
        default=215,
        type=int,
        help='gray threthold for trimed image in recognition text'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = args_parse()

    img_dir_path = os.path.join(BASE_PATH, args.img_dir)

    img_path_list = glob.glob(img_dir_path + '*.jpg')
    img_path_list += glob.glob(img_dir_path + '*.png')

    if args.make_data:
        for img_path in tqdm(img_path_list):
            monitor_img, img_name = detect_monitor.process(img_path, args)
            polys, clipped_imgs = detect_text.get_gui_result(monitor_img, img_name, args)

        print('Making data is done\nNextly, create dataset dir')

    else:
        result2table = recog_text.Result2Table(args)
        polys_list = []
        clipped_imgs_list = []
        img_name_list = []

        for img_path in tqdm(img_path_list):
            monitor_img, img_name = detect_monitor.process(img_path, args)
            polys, clipped_imgs = detect_text.get_gui_result(monitor_img, img_name, args)
            table = result2table.get_table(polys, clipped_imgs, img_name)
