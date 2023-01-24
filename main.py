import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts/craft_pytorch'))
from scripts import clip_monitor
from scripts import run_craft as craft

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--img_dir',
        default='imgs/',
        type=str,
        help='folder path to input images'
    )
    parser.add_argument(
        '--result_dir',
        default='result/',
        type=str,
        help='folder path to result'
    )

    # for monitor detection
    parser.add_argument(
        '--min_gray',
        default=100,
        type=int,
        help='min gray threthold for image processing'
    )
    parser.add_argument(
        '--max_gray',
        default=255,
        type=int,
        help='max gray threthold for image processing'
    )
    parser.add_argument(
        '--min_area',
        default=50000,
        type=int,
        help='min area threthold for image processing'
    )
    parser.add_argument(
        '--max_area',
        default=1000000,
        type=int,
        help='max area threthold for image processing'
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

    # for CRAFT
    parser.add_argument(
        '--model_dir',
        default='model',
        type=str,
        help='CRAFT model dir'
    )
    parser.add_argument(
        '--text_threshold',
        default=0.7,
        type=float,
        help='CRAFT text confidence threshold'
    )
    parser.add_argument(
        '--low_text',
        default=0.4,
        type=float,
        help='CRAFT text low-bound score'
    )
    parser.add_argument(
        '--link_threshold',
        default=0.4,
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

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = args_parse()

    img_dir_path = os.path.join(BASE_PATH, args.img_dir)
    result_dir_path = os.path.join(BASE_PATH, args.result_dir)
    Path(result_dir_path).mkdir(parents=True, exist_ok=True)

    monitor_img_list = clip_monitor.process(img_dir_path, result_dir_path, args)

    craft_imgs = craft.main(BASE_PATH, result_dir_path, args)
