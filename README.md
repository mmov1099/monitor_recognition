# Setup
```bash
git clone https://github.com/mmov1099/monitor_recognition.git
cd monitor_recognition
pip install -r requirements.txt
```
If an error occurs during installation pytorch

Please adapt to your environment　[参考](https://pytorch.org/get-started/locally/)
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
# Usage
```bash
pwd # /monitor_recognition
```
Bring monitor images to `monitor_recognition/imgs/`.
```python
python main.py
```
## options
```bash
python main.py -h
```
```bash
usage: main.py [-h] [--img_dir IMG_DIR] [--result_dir RESULT_DIR] [--min_gray MIN_GRAY] [--max_gray MAX_GRAY] [--min_area MIN_AREA]
               [--max_area MAX_AREA] [--save_contours] [--save_monitor] [--model_dir MODEL_DIR] [--text_threshold TEXT_THRESHOLD]
               [--low_text LOW_TEXT] [--link_threshold LINK_THRESHOLD] [--cuda] [--canvas_size CANVAS_SIZE] [--mag_ratio MAG_RATIO]
               [--poly] [--show_time] [--save_txt] [--save_craft]

optional arguments:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     folder path to input images
  --result_dir RESULT_DIR
                        folder path to result
  --min_gray MIN_GRAY   min gray threthold for image processing
  --max_gray MAX_GRAY   max gray threthold for image processing
  --min_area MIN_AREA   min area threthold for image processing
  --max_area MAX_AREA   max area threthold for image processing
  --save_contours       save contours of monitor in result dir
  --save_monitor        save clipped monitor img in result dir
  --model_dir MODEL_DIR
                        CRAFT model dir
  --text_threshold TEXT_THRESHOLD
                        CRAFT text confidence threshold
  --low_text LOW_TEXT   CRAFT text low-bound score
  --link_threshold LINK_THRESHOLD
                        CRAFT link confidence threshold
  --cuda                Use cuda for inference for CRAFT
  --canvas_size CANVAS_SIZE
                        CRAFT image size for inference
  --mag_ratio MAG_RATIO
                        image magnification ratio
  --poly                enable polygon type
  --show_time           show processing time
  --save_txt            save craft result as txt file
  --save_craft          save craft result as jpg file
```
