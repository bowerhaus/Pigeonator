
# Derived from motion_detection.py from the TensorFlow examples at:
# https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/

    # Portions Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# This derived code by (c) 2020 Andy Bower and is MIT Licensed

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import re
import time
import gc

from gpiozero import CPUTemperature
from annotation import Annotator

import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 1232
CAMERA_HEIGHT = 1232

THROTTLE_TEMP = 80
THROTTLE_SLEEP = 5

PHOTO_WAIT_PERIOD = 2

DEFAULT_THRESHOLD = 0.40

IGNORE_LIST = ['potted plant', 'bench', 'person', 'car']

class SmartMotionDetector():
  
    def __init__(self, args):
        self.labels = self.load_labels(args.labels)
        self.threshold = args.threshold
        self.last_signature = ""
        self.first_time = True
        self.interpreter = Interpreter(args.model)
        self.interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = self.interpreter.get_input_details()[0]['shape']

    def load_labels(self, path):
        """
        Loads the labels file. Supports files with or without index numbers.
        """
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
        """
        Sets the input tensor.
        """
        tensor_index = self.interpreter.get_input_details()[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        """
        Returns the output tensor at the given index.
        """
        output_details = self.interpreter.get_output_details()[index]
        tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        return tensor

    def detect_objects(self, image):
        """
        Returns a list of detection results, each a dictionary of object info.
        """
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
        """
        Draws the bounding box and label for each object in the results.
        """
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
        """
        When the detection results are deemed to contain something
        of interest we process them here.
        """
        for each in results:
            label = self.labels[each["class_id"]]
            print("Found %s with score %0.2f, temp=%0.1fC" % (label, each["score"], self.cpu.temperature))                   
    
        class0 = results[0]["class_id"]
        label0 = self.labels[class0]
        score0 = results[0]["score"]
        datestr = time.strftime("%Y%m%d")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        
        directory = "images/%s" % datestr
        if not os.path.isdir(directory):
            os.mkdir(directory)
            
        filename = "%s/%s-%s(%d%%).jpg" % (directory, timestr, label0, score0*100)
        print("Saving image as %s" % filename)
        image.save(filename)
                  
        time.sleep(PHOTO_WAIT_PERIOD)
    

    def check_cpu_temperature(self):
        """
        Check the CPU temperature and wait for cool down if necessary.
        """

        while (self.cpu.temperature >= THROTTLE_TEMP):
            print("Over-temperature throttling (%0.1fC)..." % self.cpu.temperature)
            time.sleep(THROTTLE_SLEEP)
            self.cpu = CPUTemperature()
     
    def get_single_signature(self, result):
        """
        Answer a signature string for a single result. The signature consists of a class
        followed by the location rounded to a 10x10 grid to avoid the detection of small
        movements.
        """
        ymin, xmin, ymax, xmax = result['bounding_box']
        cx = round((xmax+xmin) / 2, 1)
        cy = round((xmax+xmin) / 2, 1)
        signature = "%s(%d,%d)" % (self.labels[result['class_id']], cx*10, cy*10)
        return signature
    

    def get_full_signature(self, results):
        """
        Answer a signature string computed for all the give detection results.
        """
        signature = ""
        for each in results:
            eachsig = self.get_single_signature(each)
            signature = signature + eachsig + "/"
        return signature
        
    def detect_loop(self):
        """
        Loops taking a photo and submitting it to the Tensorflow engine to detect objects. The results
        are filtered to only include objects over a given threshold and to ignore classes that are deemed
        as unwanted.
        """
        gc.collect()    
        self.stream = io.BytesIO()
        self.annotator = Annotator(self.camera)
      
        for _ in self.camera.capture_continuous(self.stream, format='jpeg', use_video_port=True):
            self.stream.seek(0)
            image = Image.open(self.stream).convert('RGB')
            imageForDetect = image.resize((self.input_width, self.input_height), Image.ANTIALIAS)
    
            if self.first_time:
                imageForDetect.save("images/start.jpg")
                self.first_time = False
    
            start_time = time.monotonic()
            results = self.detect_objects(imageForDetect)
            elapsed_ms = (time.monotonic() - start_time) * 1000
    
            # Filter results over threshold
            results = [item for item in results if item["score"] >= self.threshold]
            
            # Filter out ignores
            results = [item for item in results if not self.labels[item["class_id"]] in IGNORE_LIST]
            
            # Now compute a "signature" for these significant results. We'll compare this with
            # the last one to see if there has been motion.
            self.cpu = CPUTemperature()
            signature = self.get_full_signature(results)
            if signature != self.last_signature and signature != "":
                # There has been a change
                print("Motion detected as %s" % signature)
                self.on_detect(results, image)
                self.last_signature = signature

            self.annotator.clear()
            self.annotate_objects(results)
            self.annotator.text([5, 0], '%.1fms' % (elapsed_ms))
            self.annotator.update()
            
            self.stream.seek(0)
            self.stream.truncate()         
            self.check_cpu_temperature()

    def run(self):
        print("Starting Detecton")
        with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as self.camera:
            try:
                #self.camera.exposure_mode = 'off'
                #self.camera.shutter_speed = 4000
                self.camera.start_preview()
                results, image = self.detect_loop()
                    
            finally:
                self.camera.stop_preview()
        print("Ending Detection")
        
def get_args():
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
    return parser.parse_args()
            
def main():
    args = get_args()
    detector = SmartMotionDetector(args)
    detector.run()
    
if __name__ == '__main__':
  main()

