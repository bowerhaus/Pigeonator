import math
import time
from datetime import datetime
from PIL import Image

class CameraScanner():
    def __init__(self, getfromcamera, config):
        self.getfromcamera = getfromcamera
        self.camera_width = config["camera"]["width"].get(int)
        self.camera_height = config["camera"]["height"].get(int)
        self.segment_rows = config["scanner"]["rows"].get(int)
        self.segment_cols = config["scanner"]["cols"].get(int)
        segment_ceilwidth = math.ceil(self.camera_width / self.segment_cols)
        segment_ceilheight = math.ceil(self.camera_height / self.segment_rows)
        self.segment_size = math.floor(min(segment_ceilheight, segment_ceilwidth) * config["scanner"]["segment_overlap"].get())
        self.segment_size = config["scanner"]["segment_size"].get()
        self.frame_image = None

    def get_frame_image(self):
        return self.frame_image
    
    def get_segment_image(self, n):
        return self.segment_crop(n)

    def get_next_frame(self):
        self.frame_image = self.getfromcamera()

    def segment_crop(self, n):
        image_width, image_height = self.frame_image.size
        segment_floorwidth = image_width // self.segment_cols
        segment_floorheight = image_height // self.segment_rows
        segment_col = n % self.segment_cols
        segment_row = n // self.segment_cols

        # Compute crop area
        left = segment_floorwidth*segment_col
        right = left+self.segment_size-1
        if (right > image_width):
            overlap = right-image_width
            left -= overlap
            right -= overlap
        top = segment_floorheight*segment_row
        bottom = top+self.segment_size-1
        if (bottom > image_height):
            overlap = bottom-image_height
            top -= overlap
            bottom -= overlap

        # Crop it
        cropped_image = self.frame_image.crop((left, top, right, bottom))
        cropped_image = cropped_image.resize((self.segment_size, self.segment_size), Image.ANTIALIAS)
        return cropped_image

