# img_viewer.py

import PySimpleGUI as sg
import os.path
import io
import picamera
import LobeClassifier

from PIL import Image
from datetime import datetime
from CameraScanner import CameraScanner

CAMERA_WIDTH = 4056
CAMERA_HEIGHT = 3040

SEGMENT_COLS = 3
SEGMENT_ROWS = 2
SEGMENT_SIZE = 1024

INPUT_WIDTH = 300
INPUT_HEIGHT = 300

# First the window layout in 2 columns

class PigeonatorUI:
    def __init__(self):

        frame_image_column = [
            [sg.Image(key="-IMAGE-", size=(660,660), enable_events=True)],
            [sg.Image(key="-IMAGE0-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE1-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE2-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE3-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE4-", size=(100,100), enable_events=True),
            sg.Image(key="-IMAGE5-", size=(100,100), enable_events=True)]
        ]

        # ----- Full layout -----
        layout = [
            [
                sg.Column(frame_image_column),
            ]
        ]   

        self.camera = picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=15)
        self.stream = io.BytesIO()
        self.scanner = CameraScanner(lambda: self.get_camera_image(), 0, CAMERA_WIDTH, CAMERA_HEIGHT, SEGMENT_COLS, SEGMENT_ROWS, SEGMENT_SIZE)
        self.window = sg.Window("Pigeonator UI", layout)
        self.classifier= LobeClassifier.Classifier("http://192.168.0.82:38100/predict/3290603c-b7d1-4532-8834-6713549008de")

    def get_camera_image(self):
        self.stream.seek(0)
        image = Image.open(self.stream).convert('RGB')
        self.stream.seek(0)
        self.stream.truncate()   
        return image

    def run(self):
        # Run the Event Loop

        for _ in self.camera.capture_continuous(self.stream, format='jpeg', resize=None, use_video_port=True):
            event, values = self.window.read(timeout=100)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            if event == "__TIMEOUT__":  # A file was chosen from the listbox
                image = self.scanner.get_next_image()
                if (self.scanner.segment == 0):
                    frame_image = self.scanner.get_frame_image()
                    frame_image = frame_image.resize((660, 660))
                    bio = io.BytesIO()
                    frame_image.save(bio, format="PNG")
                    self.window["-IMAGE-"].update(data=bio.getvalue())

                image = image.resize((100,100))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                self.window[f"-IMAGE{self.scanner.segment}-"].update(data=bio.getvalue())

            for i in range(0,6):
              if event == f"-IMAGE{i}-":
                segment_image = self.scanner.get_segment_image(i)
                imageForClassify = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.ANTIALIAS)
                prediction = self.classifier.get_prediction(imageForClassify)
                label = prediction["Prediction"][0]
                confidence=0
                
                # Save the image out in a batched/classified (possible incorrectly) directory
                now = datetime.now()
                batch = now.strftime("%H00H")
                batch_dir = f"images/{label}/{batch}/{self.scanner.segment}"
                if (not os.path.isdir(batch_dir)):
                    os.makedirs(batch_dir)
                save_file = f"{batch_dir}/{self.scanner.long_filename()}"
                image.save(save_file)

                print(f"Found {label} @ {confidence} - {save_file}")

        self.window.close()

def main():
    ui = PigeonatorUI()
    ui.run()
    
if __name__ == '__main__':
  main()