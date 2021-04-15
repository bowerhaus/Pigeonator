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

CAMERA_WIDTH = 300
CAMERA_HEIGHT = 300

THROTTLE_TEMP = 80
THROTTLE_SLEEP = 5

class Pigeonator():
    def __init__(self, args):
        self.show_overlay=args.overlay
        self.display_result=args.display
        self.frame_delay=int(args.delay)
        
        if (args.uselobe):
            self.classifier= LobeClassifier.Classifier(args)
        else:        
            self.classifier = TFLiteClassifier.Classifier(args)
            
        self.input_width = CAMERA_WIDTH
        self.input_height = CAMERA_HEIGHT
        
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

      
    def detect_loop(self):
        """
        Loops taking a photo and submitting it to the Tensorflow engine to classify. The results
        are filtered to only include objects over a given threshold and to ignore classes that are deemed
        as unwanted.
        """
        gc.collect()    
        self.stream = io.BytesIO()
        start_time = time.monotonic()
      
        for _ in self.camera.capture_continuous(self.stream, format='jpeg', use_video_port=True):
            self.stream.seek(0)
            image = Image.open(self.stream).convert('RGB')
            imageForDetect = image.resize((self.input_width, self.input_height), Image.ANTIALIAS)
                
            imageForDetect.save("images/im.jpg")
            self.cpu = CPUTemperature()
            temp = round(self.cpu.temperature,1)
            
            frame_ms = int((time.monotonic() - start_time) * 1000)
            start_time = time.monotonic()
                 
            prediction = self.classifier.get_prediction(imageForDetect);
            #label = prediction["Prediction"]
            #confidence = round(prediction["Confidences"][1],2)
            label = prediction["Prediction"][0]
            confidence=0
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            
            print(f"Found {label} @ {confidence} - {temp}C")
            self.camera.annotate_background = picamera.Color('black')
            self.camera.annotate_text = f"{label}@{confidence}\n{elapsed_ms}ms\n{frame_ms}ms"
            self.camera.annotate_text_size=20
#             imageForDetect.save(f"images/{label}.jpg")

            # imageForDetect.show()
            
            if (self.display_result):
                self.show_image(label+".png")

            self.stream.seek(0)
            self.stream.truncate()         
            self.check_cpu_temperature()
            
            time.sleep(self.frame_delay/1000)

    def run(self):
        print("Starting Detecton")
        with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as self.camera:
            try:
#                 self.camera.exposure_mode = 'off'
#                 self.camera.shutter_speed = 40000
                if (self.show_overlay):
                    self.camera.start_preview()
                results, image = self.detect_loop()

                    
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

