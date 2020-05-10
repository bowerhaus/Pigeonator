# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time
import gc

from gpiozero import CPUTemperature
from annotation import Annotator

import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 1640
CAMERA_HEIGHT = 1232

THROTTLE_TEMP = 80
THROTTLE_SLEEP = 5

PHOTO_COUNT = 5
PHOTO_INTERVAL = 1
PHOTO_WAIT_PERIOD = 10

DEFAULT_THRESHOLD = 0.5

class SmartDetector():
  
    def __init__(self, args):
        self.labels = self.load_labels(args.labels)
        self.threshold = args.threshold
        self.interpreter = Interpreter(args.model)
        self.interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = self.interpreter.get_input_details()[0]['shape']

    def load_labels(self, path):
        """Loads the labels file. Supports files with or without index numbers."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()              
                else:
                    labels[row_number] = pair[0].strip()
            return labels

    def set_input_tensor(self, image):
        """Sets the input tensor."""
        tensor_index = self.interpreter.get_input_details()[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        """Returns the output tensor at the given index."""
        output_details = self.interpreter.get_output_details()[index]
        tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        return tensor

    def detect_objects(self, image):
        """Returns a list of detection results, each a dictionary of object info."""
        self.set_input_tensor(image)
        self.interpreter.invoke()

        # Get all output details
        boxes = self.get_output_tensor(0)
        classes = self.get_output_tensor(1)
        scores = self.get_output_tensor(2)
        count = int(self.get_output_tensor(3))

        results = []
        for i in range(count):
            if scores[i] >= self.threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                    }
                results.append(result)
        return results

    def annotate_objects(self, results):
        # Draws the bounding box and label for each object in the results.
        for obj in results:
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * CAMERA_WIDTH)
            xmax = int(xmax * CAMERA_WIDTH)
            ymin = int(ymin * CAMERA_HEIGHT)
            ymax = int(ymax * CAMERA_HEIGHT)

            # Overlay the box, label, and score on the camera preview
            self.annotator.bounding_box([xmin, ymin, xmax, ymax])
            self.annotator.text([xmin, ymin],
                   '%s\n%.2f' % (self.labels[obj['class_id']], obj['score']))

    def get_score(self, item):
        return item["score"]
    
    def on_detect(self, results, image):
        for each in results:
            label = self.labels[each["class_id"]]
            print("Found %s with score %0.2f, temp=%0.1fC" % (label, each["score"], self.cpu.temperature))                   
    
        class0 = results[0]["class_id"]
        label0 = self.labels[class0]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "images/%s-%s.jpg" % (timestr, label0)
        print("Saving image as %s" % filename)
        image.save(filename)
        
        """Now loop and take photo at intervals"""
        """for i in range(1, PHOTO_COUNT):
            time.sleep(PHOTO_INTERVAL)
            
            self.stream.seek(0)
            image2 = Image.open(self.stream).convert('RGB')
            filename = "images/%s-%s-%d.jpg" % (timestr, label0, i)
            print("Saving image as %s" % filename)
            image2.save(filename)
            
            self.stream.seek(0)
            self.stream.truncate()"""
            
        time.sleep(PHOTO_WAIT_PERIOD)

    def detect_loop(self):
        gc.collect()    
        self.stream = io.BytesIO()
        self.annotator = Annotator(self.camera)
      
        for _ in self.camera.capture_continuous(self.stream, format='jpeg', use_video_port=True):
            self.stream.seek(0)
            image = Image.open(self.stream).convert('RGB')
            imageForDetect = image.resize((self.input_width, self.input_height), Image.ANTIALIAS)
    
            start_time = time.monotonic()
            results = self.detect_objects(imageForDetect)
            elapsed_ms = (time.monotonic() - start_time) * 1000
    
            results.sort(key=self.get_score, reverse=True)

            self.annotator.clear()
            self.annotate_objects(results)
            self.annotator.text([5, 0], '%.1fms' % (elapsed_ms))
            self.annotator.update()
            
            self.cpu = CPUTemperature()
            if len(results)>0:
                self.on_detect(results, image)
            else:
                print("Nothing to report. Temp = %0.1fC" % self.cpu.temperature)
    
            self.stream.seek(0)
            self.stream.truncate()
            
            if (self.cpu.temperature >= THROTTLE_TEMP):
                print("Temperature throttling at: %0.1fC" % self.cpu.temperature)
                time.sleep(THROTTLE_SLEEP)

    def run(self):
        with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as self.camera:
            try:
                self.camera.start_preview()
                results, image = self.detect_loop()
                    
            finally:
                self.camera.stop_preview()
            
def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
      '--model', help='File path of .tflite file.',
      required=False,
      default="models/detect.tflite")
    parser.add_argument(
      '--labels', help='File path of labels file.',
      required=False,
      default="models/coco_labels.txt")
    parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      required=False,
      type=float,
      default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    
    detector = SmartDetector(args)
    detector.run()
    
if __name__ == '__main__':
  main()

