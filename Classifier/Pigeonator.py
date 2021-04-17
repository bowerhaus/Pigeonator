import argparse
import io
import os
import re
import time
import gc
import ST7735
import picamera
import TFLiteClassifier
import LobeClassifier

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
from gpiozero import CPUTemperature

CAMERA_WIDTH = 4056
CAMERA_HEIGHT = 3040

INPUT_WIDTH = 300
INPUT_HEIGHT = 300

THROTTLE_TEMP = 80
THROTTLE_SLEEP = 5

class CameraScanner():
    def __init__(self, getfromcamera, cols=4, rows=3):
        self.getfromcamera = getfromcamera
        self.rows = rows
        self.cols = cols
        self.segment = -1
        self.image = None

    def get_next_image(self):
        self.segment = (self.segment+1) % (self.rows*self.cols)
        if (self.segment == 0):
            self.image = self.getfromcamera()
            self.image.save("images/im.jpg")
        return self.segment_crop()

    def segment_crop(self):
        image_width, image_height = self.image.size
        segment_width = image_width // self.cols
        segment_height = image_height // self.rows
        segment_col = self.segment % self.cols
        segment_row = self.segment // self.cols
        left = segment_width*segment_col
        top = segment_height*segment_row
        right = left+segment_width-1
        bottom = top+segment_height-1
        cropped_image = self.image.crop((left, top, right, bottom))
        cropped_image.save("images/im{0}.jpg".format(self.segment))
        return cropped_image

class Pigeonator():
    def __init__(self, args):
        self.show_overlay=args.overlay
        self.display_result=args.display
        self.frame_delay=int(args.delay)
        self.scanner = CameraScanner(lambda: self.get_camera_image())
        
        if (args.uselobe):
            self.classifier= LobeClassifier.Classifier(args)
        else:        
            self.classifier = TFLiteClassifier.Classifier(args)
            
        self.input_width = INPUT_WIDTH
        self.input_height = INPUT_HEIGHT
        
        if (self.display_result):
            self.init_display()
     
    def init_display(self):
        self.display = ST7735.ST7735(
        port=0,
        cs=ST7735.BG_SPI_CS_FRONT,  # BG_SPI_CSB_BACK or BG_SPI_CS_FRONT
        dc=9,
        backlight=19,               # 18 for back BG slot, 19 for front BG slot.
        spi_speed_hz=4000000)
        self.display.begin()
        
    def show_image(self, name):
        img = Image.open(name)
        self.display.display(img)   

    def check_cpu_temperature(self):
        """
        Check the CPU temperature and wait for cool down if necessary.
        """
        while (self.cpu.temperature >= THROTTLE_TEMP):
            print("Over-temperature throttling (%0.1fC)..." % self.cpu.temperature)
            time.sleep(THROTTLE_SLEEP)
            self.cpu = CPUTemperature()

    def get_camera_image(self):
        self.stream.seek(0)
        image = Image.open(self.stream).convert('RGB')
        self.stream.truncate()   
        return image

    def detect_loop(self):
        """
        Loops taking a photo and submitting it to the Tensorflow engine to classify. The results
        are filtered to only include objects over a given threshold and to ignore classes that are deemed
        as unwanted.
        """
        gc.collect()    
        self.stream = io.BytesIO()
        start_time = time.monotonic()
      
        for _ in self.camera.capture_continuous(self.stream, format='jpeg', resize=None, use_video_port=False):
            image = self.scanner.get_next_image()
            imageForDetect = image.resize((self.input_width, self.input_height), Image.ANTIALIAS)
                
            self.cpu = CPUTemperature()
            temp = round(self.cpu.temperature,1)
            
            frame_ms = int((time.monotonic() - start_time) * 1000)
            start_time = time.monotonic()
                 
            prediction = self.classifier.get_prediction(imageForDetect)
            #label = prediction["Prediction"]
            #confidence = round(prediction["Confidences"][1],2)
            label = prediction["Prediction"][0]
            confidence=0
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            
            save_file = f"images/{label}/im{self.scanner.segment}.jpg"
            print(f"Found {label} @ {confidence} - {save_file} - {temp}C")
            image.save(save_file)

            self.camera.annotate_background = picamera.Color('black')
            self.camera.annotate_text = f"{label}@{confidence}\n{elapsed_ms}ms\n{frame_ms}ms"
            self.camera.annotate_text_size=20

            # imageForDetect.show()
            
            if (self.display_result):
                self.show_image(label+".png")
      
            self.check_cpu_temperature()          
            time.sleep(self.frame_delay/1000)

    def run(self):
        print("Starting Detecton")
        with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=15) as self.camera:
            try:
                self.camera.exposure_mode = 'off'
                if (self.show_overlay):
                    self.camera.start_preview()
                self.detect_loop()

                    
            finally:
                self.camera.stop_preview()
        print("Ending Detection")
        
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
      '-model', help='Folder for model & signature file',
      required=False,
      default="models/Pigeonator2-Accuracy TFLite")
    parser.add_argument(
      '-lobeapi', help='Url for Lobe API',
      required=False,
      default="http://192.168.0.82:38100/predict/3290603c-b7d1-4532-8834-6713549008de")
    parser.add_argument(
      '-overlay', help='Display the live camera overlay',
      required=False,
      default=False,
      action='store_true')
    parser.add_argument(
      '-display', help='Display the classification result on the mini screen',
      required=False,
      default=False,
      action='store_true')
    parser.add_argument(
      '-uselobe', help='Use Lobe via the remote API',
      required=False,
      default=False,
      action='store_true')
    parser.add_argument(
      '-delay', help='Delay between frames in ms',
      required=False,
      default=0)
    return parser.parse_args()
            
def main():
    pigeonator = Pigeonator(get_args())
    pigeonator.run()
    
if __name__ == '__main__':
  main()

