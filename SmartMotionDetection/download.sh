#!/bin/bash

# Install required Python packages
python3 -m pip install -r packages.txt

# Now the TF Lite runtime only
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl

# Get TF Lite model and labels
curl -O http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d models
rm coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

# Get a labels file with corrected indices, delete the other one
(cd models && curl -O  https://dl.google.com/coral/canned_models/coco_labels.txt)
rm models/labelmap.txt

echo -e "Files downloaded to models folder"