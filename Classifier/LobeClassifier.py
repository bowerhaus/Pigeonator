#URL http://localhost:38100/predict/499c264a-b402-47c1-b648-6f7063aab5c7

import json
import io
import base64
import requests
import logging

class Classifier():
    def __init__(self, endpoint):
        self.endpoint = endpoint

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
        try:
            response = requests.request("POST", self.endpoint, data=payload)
            if response.reason == "OK":
                result = json.loads(response.text)
                outputs = result['outputs']
                # print(response.text)
            else:
                print(response.reason)
                outputs = None
            return outputs
        except:
            logging.error("Could not contact: {endpoint}", endpoint=self.endpoint)
            print(f"Could not contact: {self.endpoint}")
            return None




