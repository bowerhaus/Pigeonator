import PySimpleGUI as sg
import os.path
import io
import time
import picamera
import RemoteDetector
import LinkTap
import requests
import base64
import json
import logging
import logging.config
import time
import seqlog
import importlib

from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from datetime import datetime
from gpiozero import CPUTemperature
from ifttt_webhook import IftttWebhook
from LocalConfiguration import *

seqlog.log_to_seq(
   server_url=Config["seq"]["url"].get(),
   api_key=Config["seq"]["api_key"].get(),
   level=logging.DEBUG,
   batch_size=10,
   auto_flush_timeout=10,  # seconds
   override_root_logger=True,
   json_encoder_class=json.encoder.JSONEncoder  # Optional; only specify this if you want to use a custom JSON encoder
)

class PigeonatorDetectorUI:
    def __init__(self):

        frame_image_column = [
            [sg.Image(key="-IMAGE-", size=(660,660), enable_events=True)],
            [sg.Text("Exposure:"), sg.Combo(key="-EXPOSURE-", size=(12, 1), values=["Auto", "12000", "8000", "4000", "2000", "1500", "1000", "500"], default_value="Auto", readonly=True, enable_events=True),           
            sg.Text("Contrast:"), sg.Slider(key="-CONTRAST-", orientation="h", range=(-20, 20), size=(12, 12), disable_number_display=True, default_value=0, resolution=5, tooltip=0, enable_events =True)],
            [sg.Text("Detection:"), sg.Text(key="-DETECTION-", size=(24, 1)),
            sg.Checkbox("Auto Detect", key="-DETECT-", size=(12,1), enable_events=True, default=True),
            sg.Checkbox("Deter", key="-DETER-", size=(5,1), enable_events=True, default=True),
            sg.Text(key="-FRAMETIME-", size=(15, 1), justification="right"),
            sg.Text(key="-ELAPSED-", size=(10, 1), justification="right"),
            sg.Text(key="-TEMP-", size=(5, 1), justification="right")]]

        # ----- Full layout -----
        layout = [
            [
                sg.Column(frame_image_column),
            ]
        ]  

        self.camera = picamera.PiCamera(resolution=(Config["camera"]["width"].get(), Config["camera"]["height"].get()), framerate=15)
        self.camera.iso = 100
        self.stream = io.BytesIO()
        self.confidence_threshold = Config["model"]["confidence_threshold"].get()
       
        self.window = sg.Window("Pigeonator Detector UI", layout)
        self.linktap = LinkTap.LinkTap(Config["linktap"]["username"].get(), Config["linktap"]["api_key"].get())
        self.font= ImageFont.truetype('/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf', 80)
    

    def get_camera_image(self):
        self.stream.seek(0)
        image = Image.open(self.stream).convert('RGB')
        self.stream.seek(0)
        self.stream.truncate()   
        return image

    def get_exposure(self):
        return self.values["-EXPOSURE-"]

    def get_contrast(self):
        return int(self.values["-CONTRAST-"])

    def get_detect_mode(self):
        return self.values["-DETECT-"]

    def get_deter_mode(self):
        return self.values["-DETER-"]

    def set_detection(self, text):
        self.window["-DETECTION-"].update(text)
        self.window.refresh()

    def set_temperature_display(self, text, color):
        self.window["-TEMP-"].update(text, text_color=color)
        self.window.refresh()

    def set_elapsed_display(self, time):
        self.window["-ELAPSED-"].update(f"{time}ms")
        self.window.refresh()

    def set_frametime_display(self, time):
        self.window["-FRAMETIME-"].update(f"{time}ms")
        self.window.refresh()

    def set_display_image(self, image):
        view = image.resize((660, 660))
        bio = io.BytesIO()
        view.save(bio, format="PNG")
        self.window[f"-IMAGE-"].update(data=bio.getvalue())
        self.window.refresh()

    def set_camera_exposure(self, exposure):
        if exposure == "Auto":
            self.camera.exposure_mode = 'auto'
        else:
            self.camera.exposure_mode = 'off'
            self.camera.shutter_speed = int(exposure)


    def set_camera_contrast(self, contrast):
        self.camera.contrast = contrast
        self.window["-CONTRAST-"].SetTooltip(str(contrast))


    def detector(self):
        detector_name = Config["detector"]["name"].get()
        detector_url = Config["detector"]["url"].get()
        module = importlib.import_module(detector_name)
        clss = getattr(module, detector_name)
        return clss(detector_url)

    def check_cpu_temperature(self):
        """
        Check the CPU temperature and wait for cool down if necessary.
        """
        self.cpu = CPUTemperature()
        self.temperature = round(self.cpu.temperature, 1)

        while (self.cpu.temperature >= Config["cpu"]["throttle_temp"].get(int)):
            logging.warning("Over-temperature throttling ({temp}C)...", temp=self.temperature)
            print(f"Over-temperature throttling ({self.temperature}C)...")
            self.set_temperature_display(f"{self.temperature}C", "red")
            time.sleep(Config["cpu"]["throttle_sleep"].get(int))
            self.cpu = CPUTemperature()
            self.temperature = round(self.cpu.temperature, 1)
        logging.debug("System temperature {temp}C", temp=self.temperature)
        self.set_temperature_display(f"{self.temperature}C", "white")

    def image_to_base64(self, image):
        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format = "PNG")
        
        # reset file pointer to start
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()
        return base64.b64encode(img_bytes).decode('ascii')

    def fire_sprinkler(self, secs, label):
        try:
            secs = max(secs, 3)
            logging.info(f"Deterring {label} with sprinkler for {secs} seconds", label=label)
            print(f"Deterring {label} with sprinkler for {secs} seconds")
            self.linktap.activate_instant_mode(Config["linktap"]["gateway_id"].get(), Config["linktap"]["taplinker_id"].get(), True, 0, secs, False)
            time.sleep(secs)
        
        except LinkTap.LinkTapError as error:
            logging.error(f"Failed to execute linktap command: {error.message}")
            print(f"Failed to execute linktap command: {error.message}")

    def imgbb_upload(self, image, label, description):
        payload = {
            "key": Config["imgbb"]["api_key"].get(),
            "image": self.image_to_base64(image.resize((600,600))),
            "name": description,
            "expiration": 3600*24
        }
        if not(Config["imgbb"]["active"].get(bool)):
            return ("None", "None", "None")

        reply = requests.post(Config["imgbb"]["upload_url"].get(), payload)
        if reply.reason=="OK":
            result = json.loads(reply.content)
            data = result["data"]
            url = data["url_viewer"]
            thumb_url = data["thumb"]["url"]
            image_url = data["url"]
            logging.info("Saved IMGBB image {url} for {label}", url=image_url, label=label)
            return (url, thumb_url, image_url)

        logging.error("Unable to save IMGBB image for {label}", label=label)
        return None, 

    def save_image(self, image, dir):
        if (not os.path.isdir(dir)):
            os.makedirs(dir)

        now = datetime.now()
        date_time = now.strftime("%Y%m%d%H%M%S")
        name =  f"im-{date_time}.jpg"
        save_file = f"{dir}/{name}"

        image.save(save_file)


    def detect_image(self, image):
        imageForDetect = image.resize((Config["model"]["input_width"].get(int), Config["model"]["input_height"].get(int)), Image.ANTIALIAS)

        # Make prediction
        detector = self.detector()
        prediction = detector.get_prediction(imageForDetect)
        if (prediction == None):
            self.set_detection("ERROR")
            return None

        elapsedMs = prediction["Elapsed"]
        self.set_elapsed_display(elapsedMs)

        bestitem = None
        items = prediction["Items"]
        for item in items:
            if item["label"] == "Pigeon":
                bestitem = item
                break

        if bestitem == None:
            self.set_detection("None")
            return None

        xscale = Config["camera"]["width"].get(int) / Config["model"]["input_width"].get(int)
        yscale = Config["camera"]["height"].get(int) / Config["model"]["input_height"].get(int)
        bestbox = bestitem["box"]

        label = bestitem["label"]
        confidence = round(bestitem["score"], 4)
        box = ((int(bestbox[0]*xscale), int(bestbox[1]*yscale)), (int(bestbox[2]*xscale), int(bestbox[3]*yscale)))

        lt, rb = box
        x1, y1 = lt
        x2, y2 = rb
        x=(x1+x2)//2
        y=(y1+y2)//2
        w = x2-x1
        h=y2-y1
        area = w*h
        location = (x, y)  
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logging.info("RawDetect {label} @ {confidence} at {location} A={area}", label=label, confidence=confidence, location=location, area=area)
        raw_detect_file = open("rawdetect.csv", "a")
        raw_detect_file.write(f"{ts},{label},{confidence},{x1},{y1},{x2},{y2},{x},{y},{w},{h},{area}\n")
        raw_detect_file.close()

        return (label, confidence, box, location, area)

    def run(self):
        cycle_count = 0
        this_frame_time = last_frame_time = int(time.time()*1000)

        # Run the Event Loop
        for _ in self.camera.capture_continuous(self.stream, format='jpeg', resize=None, use_video_port=True):
            event, self.values = self.window.read(timeout=200)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            last_frame_time = this_frame_time
            this_frame_time = int(time.time()*1000)
            frame_time = this_frame_time - last_frame_time
            self.set_frametime_display(frame_time)

            if event == "__TIMEOUT__":  
                self.check_cpu_temperature()
                current_image = self.get_camera_image()

            cycle_count = cycle_count+1
            if (cycle_count <= Config["camera"]["warm_up_cycles"].get(int)):
                self.set_display_image(current_image)
                continue

            if event == "-IMAGE-" or self.get_detect_mode():
                result = self.detect_image(current_image)
                if result != None:
                    label, confidence, box, location, area= result
                    confidence = round(confidence,4)        

                    if confidence >= self.confidence_threshold:
                        logging.info("Detected {label} @ {confidence} at {location} A={area}", label=label, confidence=confidence, location=location, area=area)
                        self.set_detection(f"{label} @ {confidence}")
                        self.save_image(current_image, f"images/{label}/actual")

                        draw = ImageDraw.Draw(current_image)
                        draw.rectangle(box, outline="red", width=3)
                        draw.text((50,50), f"{label} @ {confidence} at {location}, A={area}", fill="red", font=self.font)
                        self.save_image(current_image, f"images/{label}/annotated")
                        self.set_display_image(current_image)

                        if self.get_deter_mode():
                            self.fire_sprinkler(25, label)

                        description = f"{label} @ {confidence}"
                        url, _, _ = self.imgbb_upload(current_image, label, description)
                    else:
                        self.set_detection(f"None @ {round(1-confidence,4)}")
                        
                        draw = ImageDraw.Draw(current_image)
                        draw.rectangle(box, outline="blue", width=3)
                        draw.text((50,50), f"{label} @ {confidence} at {location}, A={area}", fill="blue", font=self.font)
            else:
                time.sleep(3)

            self.set_display_image(current_image)
                    
            if event == "-EXPOSURE-":
                self.set_camera_exposure(self.get_exposure())

            if event == "-CONTRAST-":
                self.set_camera_contrast(self.get_contrast())

        self.window.close()

def main():
    ui = PigeonatorDetectorUI()
    ui.run()
    
if __name__ == '__main__':
  main()