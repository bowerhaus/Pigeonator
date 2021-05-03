from datetime import datetime

class CameraZone:
    def __init__(self, n, scanner):
        self.id = n
        self.scanner = scanner
        self.is_active = False

    def get_image(self):
        self.image = self.scanner.get_segment_image(self.id)
        return self.image

    def short_filename(self):
        return f"im{self.id}.jpg"    

    def long_filename(self):
        now = datetime.now()
        date_time = now.strftime("%Y%m%d%H%M%S")
        return f"im{self.id}-{date_time}.jpg"