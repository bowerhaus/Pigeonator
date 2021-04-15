import tflite_runtime.interpreter as tflite
import os
import json
import numpy as np

class Classifier():
    def __init__(self, args):
        self.model, self.signature = self.get_model_and_sig(args.model)
        self.interpreter = tflite.Interpreter(self.model)
        self.interpreter.allocate_tensors()
        
    def get_model_and_sig(self, model_dir):
        """Method to get name of model file"""
        with open(os.path.join(model_dir, "signature.json"), "r") as f:
            signature = json.load(f)
        model_file = os.path.join(model_dir, signature.get("filename"))
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model file does not exist")
        return model_file, signature
    
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
        Predict with the interpreter!
        """
        # Combine the information about the inputs and outputs from the signature.json file with the Interpreter runtime
        signature_inputs = self.signature.get("inputs")
        input_details = {detail.get("name"): detail for detail in self.interpreter.get_input_details()}
        model_inputs = {key: {**sig, **input_details.get(sig.get("name"))} for key, sig in signature_inputs.items()}
        signature_outputs = self.signature.get("outputs")
        output_details = {detail.get("name"): detail for detail in self.interpreter.get_output_details()}
        model_outputs = {key: {**sig, **output_details.get(sig.get("name"))} for key, sig in signature_outputs.items()}

        if "Image" not in model_inputs:
            raise ValueError("Tensorflow Lite model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")

        # process image to be compatible with the model
        input_data = self.process_image(image, model_inputs.get("Image").get("shape"))

        # set the input to run
        self.interpreter.set_tensor(model_inputs.get("Image").get("index"), input_data)
        self.interpreter.invoke()

        # grab our desired outputs from the interpreter!
        # un-batch since we ran an image with batch size of 1, and convert to normal python types with tolist()
        outputs = {key: self.interpreter.get_tensor(value.get("index")).tolist()[0] for key, value in model_outputs.items()}
        # postprocessing! convert any byte strings to normal strings with .decode()
        for key, val in outputs.items():
            if isinstance(val, bytes):
                outputs[key] = val.decode()

        return outputs


