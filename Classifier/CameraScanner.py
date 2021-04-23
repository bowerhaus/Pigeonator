import math
import time
from datetime import datetime
from PIL import Image

SEGMENT_OVERLAP = 1.2

class CameraScanner():
    def __init__(self, getfromcamera, frame_delay_ms, camera_width, camera_height, segment_cols, segment_rows, segment_size):
        self.getfromcamera = getfromcamera
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.segment_rows = segment_rows
        self.segment_cols = segment_cols
        self.frame_delay_ms = frame_delay_ms
        segment_ceilwidth = math.ceil(self.camera_width / self.segment_cols)
        segment_ceilheight = math.ceil(self.camera_height / self.segment_rows)
        self.segment_size = math.floor(min(segment_ceilheight, segment_ceilwidth) * SEGMENT_OVERLAP)
        self.segment = -1
        self.segment_size = segment_size
        self.frame_image = None

    def get_next_image(self):
        self.segment = (self.segment+1) % (self.segment_rows*self.segment_cols)
        if (self.segment == 0):
            self.frame_image = self.getfromcamera()
            self.frame_image.save("images/im.jpg")
        return self.segment_crop(self.segment)

    def get_frame_image(self):
        return self.frame_image
    
    def get_segment_image(self, n):
        return self.segment_crop(n)

    def reset(self):
        segment = -1

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
        cropped_image.save(f"images/{self.short_filename()}")
        return cropped_image

    def short_filename_for(self, n):
        return f"im{n}.jpg"    

    def short_filename(self):
        return self.short_filename_for(self.segment)

    def long_filename_for(self, n):
        now = datetime.now()
        date_time = now.strftime("%Y%m%d%H%M%S")
        return f"im{n}-{date_time}.jpg"

    def long_filename(self):
        return self.long_filename_for(self.segment)