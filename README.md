# Setup
Firstly install tesseract

Other than Ubuntu -> [参考](https://tesseract-ocr.github.io/tessdoc/Installation.html)
```bash
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

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
Bring monitor images to `monitor_recognition/imgs/`
```python
python main.py
```
## options
```bash
python main.py -h
```
```bash
usage: main.py [-h] [--img_dir IMG_DIR] [--result_dir RESULT_DIR] [--monitor_gray MONITOR_GRAY] [--min_area MIN_AREA]
               [--max_area MAX_AREA] [--save_contours] [--save_monitor] [--text_detect {craft,gui,gui_each}] [--run_gui]
               [--model_dir MODEL_DIR] [--text_threshold TEXT_THRESHOLD] [--low_text LOW_TEXT]
               [--link_threshold LINK_THRESHOLD] [--cuda] [--canvas_size CANVAS_SIZE] [--mag_ratio MAG_RATIO] [--poly]
               [--show_time] [--save_txt] [--save_craft] [--height HEIGHT] [--width WIDTH] [--row ROW] [--column COLUMN]
               [--tilt TILT] [--ocr_type {easyocr,mangaocr,tesseract,pyocr}] [--use_gray] [--recog_gray RECOG_GRAY]
               [--craft_recog]

optional arguments:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     folder path to input images
  --result_dir RESULT_DIR
                        folder path to result
  --monitor_gray MONITOR_GRAY
                        gray threthold for monitor image processing
  --min_area MIN_AREA   min area threthold for monitor image processing
  --max_area MAX_AREA   max area threthold for monitor image processing
  --save_contours       save contours of monitor in result dir
  --save_monitor        save clipped monitor img in result dir
  --text_detect {craft,gui,gui_each}
                        choose text detection type, craft is not updated
  --run_gui             run detection monitor gui forcely
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
  --height HEIGHT       default height value of each cell of monitor table
  --width WIDTH         default width value of each cell of monitor table
  --row ROW             default row number of monitor table
  --column COLUMN       default column number of monitor table
  --tilt TILT           default horizon tilt value of monitor table of cells
  --ocr_type {easyocr,mangaocr,tesseract,pyocr}
                        choose ocr library
  --use_gray            convert grayscale image from trimed image for text recognition
  --recog_gray RECOG_GRAY
                        gray threthold for trimed image
  --craft_recog         use craft after trimming image for text recognition
```
