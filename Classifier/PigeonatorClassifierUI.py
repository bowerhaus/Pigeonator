import PySimpleGUI as sg
import os.path
import io
import time
import picamera
import RemoteClassifier
import LinkTap
import requests
import base64
import json
import logging
import logging.config
import time
import seqlog
import importlib

from PIL import Image
from datetime import datetime
from gpiozero import CPUTemperature
from ifttt_webhook import IftttWebhook
from CameraScanner import CameraScanner
from CameraZone import CameraZone
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


class PigeonatorClassifierUI:
    def __init__(self):

        frame_image_column = [
            [sg.Image(key="-IMAGE-", size=(660,660), enable_events=True)],
            [sg.Text("Exposure:"), sg.Combo(key="-EXPOSURE-", size=(12, 1), values=["Auto", "12000", "8000", "4000", "2000", "1500", "1000", "500"], default_value="Auto", readonly=True, enable_events=True),           
            sg.Text("Contrast:"), sg.Slider(key="-CONTRAST-", orientation="h", range=(-20, 20), size=(12, 12), disable_number_display=True, default_value=0, resolution=5, tooltip=0, enable_events =True)],
            [sg.Image(key="-IMAGE0-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE1-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE2-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE3-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE4-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE5-", size=(100,100), enable_events=True)],
            [sg.Text("Classification:"), sg.Text(key="-CLASSIFICATION-", size=(24, 1)),
            sg.Checkbox("Auto Detect", key="-DETECT-", size=(12,1), enable_events=True),
            sg.Checkbox("Deter", key="-DETER-", size=(5,1), enable_events=True),
            sg.Text(key="-TEMP-", size=(27, 1), justification="right")]]

        # ----- Full layout -----
        layout = [
            [
                sg.Column(frame_image_column),
            ]
        ]   

        self.camera = picamera.PiCamera(resolution=(Config["camera"]["width"].get(), Config["camera"]["height"].get()), framerate=15)
        self.camera.iso = 100
        self.stream = io.BytesIO()
        self.scanner = CameraScanner(lambda: self.get_camera_image(), Config)

        self.zones = []
        for id in range(self.scanner.segment_cols*self.scanner.segment_rows):
            self.zones.append(CameraZone(id, self.scanner))
            self.zones[id].is_active = True
        self.reset_current_zone()
        
        self.window = sg.Window("Pigeonator Classifier UI", layout)
        self.linktap = LinkTap.LinkTap(Config["linktap"]["username"].get(), Config["linktap"]["api_key"].get())

    def reset_current_zone(self):
        self.current_zone = -1

    def get_camera_image(self):
        self.stream.seek(0)
        image = Image.open(self.stream).convert('RGB')
        self.stream.seek(0)
        self.stream.truncate()   
        return image

    def get_next_zone(self):
        self.current_zone = (self.current_zone+1) % len(self.zones)
        if (self.current_zone == 0):
            self.scanner.get_next_frame()
        return self.zones[self.current_zone]

    def get_exposure(self):
        return self.values["-EXPOSURE-"]

    def get_contrast(self):
        return int(self.values["-CONTRAST-"])

    def get_detect_mode(self):
        return self.values["-DETECT-"]

    def get_deter_mode(self):
        return self.values["-DETER-"]

    def set_classification(self, text):
        self.window["-CLASSIFICATION-"].update(text)
        self.window.refresh()

    def set_temperature_display(self, text, color):
        self.window["-TEMP-"].update(text, text_color=color)
        self.window.refresh()

    def classifier(self):
        classifier_name = Config["classifier"]["name"].get()
        classifier_url = Config["classifier"]["url"].get()
        module = importlib.import_module(classifier_name)
        clss = getattr(module, classifier_name)
        return clss(classifier_url)

    def classify_image(self, image, n):
        zone = self.zones[n]
        if not zone.is_active:
            logging.warning("Zone {zone} is not active so not classified", zone=n)
            return

        imageForClassify = image.resize((Config["model"]["input_width"].get(int), Config["model"]["input_height"].get(int)), Image.ANTIALIAS)

        # Make prediction
        classifier = self.classifier()
        prediction = classifier.get_prediction(imageForClassify)
        if (prediction == None):
            self.set_classification("ERROR")
            return None
        
        label = prediction["Prediction"][0]

        # Find confidence of prediction from all labels
        labels = prediction["Labels"]
        for eachlabel in labels:
            if eachlabel[0] == label:
                confidence = round(eachlabel[1], 4)
                self.set_classification(f"Zone {n} is {label} @ {confidence}")
                break

        logging.info("Classified zone {camera} as {label} @ {confidence}", label=label, camera=n, confidence=confidence)
        print(f"Classified zone {n} as {label} @ {confidence} - temp={self.cpu.temperature}C")
        return (label, confidence)

    def set_frame_image(self, image):
        view = image.resize((660, 440))
        bio = io.BytesIO()
        view.save(bio, format="PNG")
        self.window[f"-IMAGE-"].update(data=bio.getvalue())

    def set_zone_image(self, n, image):
        thumb = image.resize((100,100))
        bio = io.BytesIO()
        thumb.save(bio, format="PNG")
        self.window[f"-IMAGE{n}-"].update(data=bio.getvalue())

    def set_camera_exposure(self, exposure):
        if exposure == "Auto":
            self.camera.exposure_mode = 'auto'
        else:
            self.camera.exposure_mode = 'off'
            self.camera.shutter_speed = int(exposure)
        self.reset_current_zone()

    def set_camera_contrast(self, contrast):
        self.camera.contrast = contrast
        self.window["-CONTRAST-"].SetTooltip(str(contrast))
        self.reset_current_zone()

    def save_classified_image(self, image, n, label):
        label_dir = f"images/{label}"
        if (not os.path.isdir(label_dir)):
            os.makedirs(label_dir)
        save_file = f"{label_dir}/{self.zones[n].long_filename()}"
        image.save(save_file)

    def save_full_image(self, image):
        dir = f"images/Full"
        if (not os.path.isdir(dir)):
            os.makedirs(dir)

        now = datetime.now()
        date_time = now.strftime("%Y%m%d%H%M%S")
        save_file = f"{dir}/frame-{date_time}.jpg"
        image.save(save_file)

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
            self.linktap.activate_instant_mode(Config["linktap"]["gateway_id"].get(), Config["linktap"]["taplinker_id"].get(), True, 0, secs, False)
            print(f"Deterring {label} with sprinkler for {secs} seconds")
        except:
            logging.error("Failed to execute linktap command")
            print("Failed to execute linktap command")

    def imgbb_upload(self, image, label, description):
        payload = {
            "key": Config["imgbb"]["api_key"].get(),
            "image": self.image_to_base64(image),
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


    def run(self):
        # Run the Event Loop
        for _ in self.camera.capture_continuous(self.stream, format='jpeg', resize=None, use_video_port=True):
            event, self.values = self.window.read(timeout=500)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            if event == "__TIMEOUT__":  
                current_zone = self.get_next_zone()
                if current_zone.id == 0:
                    frame_image = self.scanner.get_frame_image()
                    self.set_frame_image(frame_image)
                    frame_image.save("images/im.jpg")
                    self.check_cpu_temperature()

                zone_image = current_zone.get_image()
                zone_image.save(f"images/im{current_zone.id}.jpg")
                self.set_zone_image(current_zone.id, zone_image)

                if self.get_detect_mode():
                    result = self.classify_image(zone_image, current_zone.id)
                    if result==None:
                        continue
                    label, confidence = result
                        
                    if label == "Pigeon" and confidence >= Config["model"]["confidence_threshold"].get(float):
                        self.save_classified_image(zone_image, current_zone.id, label)
                        description = f"{label}-{current_zone.long_filename()}"
                        url, _, _ = self.imgbb_upload(zone_image, label, description)

                        if Config["ifttt"]["active"].get(bool):
                            ifttt = IftttWebhook(Config["ifttt"]["api_key"].get())
                            ifttt.trigger("PigeonatorDetect", value1=label, value2=confidence, value3=str(current_zone.id))

                        logging.info("Detected {label} @ {confidence} in zone {zone} and saved image: {im}", label=label, confidence=confidence, im=url, zone=current_zone.id)
                        if self.get_deter_mode():
                            self.fire_sprinkler(15, label)

            if event == "-EXPOSURE-":
                self.set_camera_exposure(self.get_exposure())

            if event == "-CONTRAST-":
                self.set_camera_contrast(self.get_contrast())

            for i in range(6):
              if event == f"-IMAGE{i}-":
                self.set_classification("Working...")
                zone_image = self.zones[i].get_image()
                result = self.classify_image(zone_image, i)
                if result==None:
                    self.set_classification("Not classified")
                    break
                label, confidence = result

                if label == "Pigeon":
                    description = f"{label}-{self.zones[i].long_filename()}"
                    #url, _, _ = self.imgbb_upload(zone_image, label, description)
                    #ifttt = IftttWebhook(Secrets.IFTTT_API_KEY)
                    #ifttt.trigger("PigeonatorDetect", value1=label, value2=str(i), value3=url)

                # Save in an ALL batch
                self.save_classified_image(zone_image, i, "Training")
                self.save_full_image(self.scanner.get_frame_image())

        self.window.close()

def main():
    ui = PigeonatorClassifierUI()
    ui.run()
    
if __name__ == '__main__':
  main()