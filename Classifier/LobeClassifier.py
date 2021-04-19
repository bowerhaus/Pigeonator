#URL http://localhost:38100/predict/499c264a-b402-47c1-b648-6f7063aab5c7

import json
import io
import base64
import requests

class Classifier():
    def __init__(self, endpoint):
        self.endpoint = endpoint
    
    def process_image(self, image, input_shape):
        """
        Given a PIL Image, center square crop and resize to fit the expected model input, and convert from [0,255] to [0,1] values.
        """
        width, height = image.size
        # ensure image type is compatible with model and convert if not
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
            
        # now the image is square, resize it to be the right shape for the model input
        input_width, input_height = input_shape[1:3]

        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
        image = np.asarray(image) / 255.0
        # format input as model expects
        return image.reshape(input_shape).astype(np.float32)

    def get_prediction(self, image):
        """
        Predict with the remote classifier!
        """

        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format = "PNG")
        
        # reset file pointer to start
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()

        b64data = base64.b64encode(img_bytes).decode('ascii')
        
        payload = "{\"inputs\":{\"Image\":\"%s\"}}" % b64data
        response = requests.request("POST", self.endpoint, data=payload)
        
        result = json.loads(response.text)
        outputs = result['outputs']

        print(response.text)
        
#         input_data = self.process_image(image, model_inputs.get("Image").get("shape"))

        return outputs



