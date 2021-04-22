# img_viewer.py

import PySimpleGUI as sg
import os.path
import io
import time
import picamera
import LobeClassifier
import LinkTap
import requests
import base64
import json
import logging
import logging.config
import time
import loggly.handlers

logging.config.fileConfig('Python.conf')
logging.Formatter.converter = time.gmtime
logger = logging.getLogger('Pigeonator')

from PIL import Image
from datetime import datetime
from gpiozero import CPUTemperature
from CameraScanner import CameraScanner

CAMERA_WIDTH = 3000
CAMERA_HEIGHT = 2000

SEGMENT_COLS = 3
SEGMENT_ROWS = 2
SEGMENT_SIZE = 1000

INPUT_WIDTH = 300
INPUT_HEIGHT = 300

THROTTLE_TEMP = 70
THROTTLE_SLEEP = 10

DEFAULT_MODEL = "Pigeonator3"

LINKTAP_GATEWAY = "7A327022004B1200B69"
LINKTAP_TAPLINKER = "8F2F7022004B1200"
LINKTAP_USERNAME = "Bowerandy"
LINKTAP_APIKEY = "ed5f1074c05715e3e65723db9e321cb9"

IMGBB_UPLOAD = "https://api.imgbb.com/1/upload"
IMGBB_API_KEY = "3b43324bcbc8164bd9be44ded3651664"

# First the window layout in 2 columns

class PigeonatorUI:
    def __init__(self):
        self.models = {
            "Pigeonator2" : "3290603c-b7d1-4532-8834-6713549008de",
            "Pigeonator3" : "d06b0699-87c3-4831-a62f-99644aebff3d",
            "Pigeonator4" : "b0ae8b35-7bd5-4fb4-ab65-b20b8c177a9d"
        }
        model_names = list(self.models.keys())

        frame_image_column = [
            [sg.Image(key="-IMAGE-", size=(660,660), enable_events=True)],
            [sg.Text("Exposure:"), sg.Combo(key="-EXPOSURE-", size=(12, 1), values=["Auto", "12000", "8000", "4000", "2000", "1500", "1000", "500"], default_value="Auto", readonly=True, enable_events=True)],
            [sg.Image(key="-IMAGE0-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE1-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE2-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE3-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE4-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE5-", size=(100,100), enable_events=True)],
            [sg.Text("Classifier:"), sg.Combo(key="-CLASSIFIER-", size=(12, 1), values=model_names, default_value=DEFAULT_MODEL, readonly=True, enable_events=True),
            sg.Text("Last Classification:"), sg.Text(key="-CLASSIFICATION-", size=(18, 1)),
            sg.Checkbox("Auto Detect", key="-AUTO-", size=(15,1), enable_events=True),
            sg.Text(key="-TEMP-", size=(10, 1), justification="right")]
        ]

        # ----- Full layout -----
        layout = [
            [
                sg.Column(frame_image_column),
            ]
        ]   

        self.camera = picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=15)
        self.camera.iso = 100
        #self.camera.contrast = -10
        self.stream = io.BytesIO()
        self.scanner = CameraScanner(lambda: self.get_camera_image(), 0, CAMERA_WIDTH, CAMERA_HEIGHT, SEGMENT_COLS, SEGMENT_ROWS, SEGMENT_SIZE)
        self.window = sg.Window("Pigeonator UI", layout)
        self.linktap = LinkTap.LinkTap(LINKTAP_USERNAME, LINKTAP_APIKEY)

    def get_camera_image(self):
        self.stream.seek(0)
        image = Image.open(self.stream).convert('RGB')
        self.stream.seek(0)
        self.stream.truncate()   
        return image

    def get_exposure(self):
        return self.values["-EXPOSURE-"]

    def get_classifier_name(self):
        return self.values["-CLASSIFIER-"]

    def get_model(self):
        return self.models[self.get_classifier_name()]

    def get_auto_mode(self):
        return self.values["-AUTO-"]

    def classifier(self):
        return LobeClassifier.Classifier(f"http://192.168.0.82:38100/predict/{self.get_model()}")

    def set_classification(self, text):
        self.window["-CLASSIFICATION-"].update(text)
        self.window.refresh()

    def set_temperature_display(self, text, color):
        self.window["-TEMP-"].update(text, text_color=color)
        self.window.refresh()

    def classify_image(self, image, n):
        imageForClassify = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.ANTIALIAS)

        # Make prediction
        prediction = self.classifier().get_prediction(imageForClassify)
        if (prediction == None):
            self.set_classification("ERROR")
            return None
        
        label = prediction["Prediction"][0]

        # Find confidence of prediction from all labels
        labels = prediction["Labels"]
        for eachlabel in labels:
            if eachlabel[0] == label:
                confidence = eachlabel[1]
                self.set_classification(f"{label}({n}) @ {confidence}")
                break

        print(f"Found {label}({n}) @ {confidence} - temp={self.cpu.temperature}C")
        return (label, confidence)

    def set_frame_image(self, image):
        view = image.resize((660, 440))
        bio = io.BytesIO()
        view.save(bio, format="PNG")
        self.window[f"-IMAGE-"].update(data=bio.getvalue())

    def set_view_image(self, n, image):
        view = image.resize((100,100))
        bio = io.BytesIO()
        view.save(bio, format="PNG")
        self.window[f"-IMAGE{n}-"].update(data=bio.getvalue())

    def set_camera_exposure(self, exposure):
        if exposure == "Auto":
            self.camera.exposure_mode = 'auto'
        else:
            self.camera.exposure_mode = 'off'
            self.camera.shutter_speed = int(exposure)
        self.scanner.reset()

    def save_classified_image(self, image, n, label):
        label_dir = f"images/{label}"
        if (not os.path.isdir(label_dir)):
            os.makedirs(label_dir)
        save_file = f"{label_dir}/{self.scanner.long_filename_for(n)}"
        image.save(save_file)

    def check_cpu_temperature(self):
        """
        Check the CPU temperature and wait for cool down if necessary.
        """
        self.cpu = CPUTemperature()
        self.temperature = round(self.cpu.temperature,1)

        while (self.cpu.temperature >= THROTTLE_TEMP):
            print("Over-temperature throttling (%0.1fC)..." % self.cpu.temperature)
            self.set_temperature_display(f"{self.temperature}C", "red")
            time.sleep(THROTTLE_SLEEP)
            self.cpu = CPUTemperature()
            self.temperature = round(self.cpu.temperature, 1)
        self.set_temperature_display(f"{self.temperature}C", "white")

    def image_to_base64(self, image):
        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format = "PNG")
        
        # reset file pointer to start
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()
        return base64.b64encode(img_bytes).decode('ascii')

    def fire_sprinkler(self, secs):
        try:
            secs = max(secs, 3)
            self.linktap.activate_instant_mode(LINKTAP_GATEWAY, LINKTAP_TAPLINKER, True, 0, secs, False)
            print(f"Firing sprinkler for {secs} seconds")
        except:
            print("Failed to execute linktap command")

    def imgbb_upload(self, image, description):
        payload = {
            "key": IMGBB_API_KEY,
            "image": self.image_to_base64(image),
            "name": description,
            "expiration": 3600*24
        }
        reply = requests.post(IMGBB_UPLOAD, payload)
        if reply.reason=="OK":
            result = json.loads(reply.content)
            data = result["data"]
            url = data["url_viewer"]
            thumb_url = data["thumb"]["url"]
            image_url = data["url"]
            return (url, thumb_url, image_url)
        return None


    def run(self):
        # Run the Event Loop
        for _ in self.camera.capture_continuous(self.stream, format='jpeg', resize=None, use_video_port=True):
            event, self.values = self.window.read(timeout=500)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            if event == "__TIMEOUT__":  
                image = self.scanner.get_next_image()
                if self.scanner.segment == 0:
                    self.set_frame_image(self.scanner.get_frame_image())
                self.set_view_image(self.scanner.segment, image)

                if self.get_auto_mode():
                    result = self.classify_image(image, self.scanner.segment)
                    if result==None:
                        break
                    label, confidence = result
                        
                    if label == "Pigeon":
                        self.save_classified_image(image, self.scanner.segment, label)
                        description = f"{label}-{self.scanner.long_filename_for(self.scanner.segment)}"
                        self.imgbb_upload(image, description)
                        self.fire_sprinkler(15)

            if event == "-EXPOSURE-":
                self.set_camera_exposure(self.get_exposure())

            for i in range(0,6):
              if event == f"-IMAGE{i}-":
                self.set_classification("Working...")
                segment_image = self.scanner.get_segment_image(i)
                result = self.classify_image(segment_image, i)
                if result==None:
                    break
                label, confidence = result

                # Save in an ALL batch
                self.save_classified_image(segment_image, i, "Training")
            self.check_cpu_temperature()

        self.window.close()

def main():
    ui = PigeonatorUI()
    ui.run()
    
if __name__ == '__main__':
  main()