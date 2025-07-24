from torchvision.transforms import ToTensor
import sys
from pathlib import Path
import torch
import json
import base64 
from PIL import Image
from io import BytesIO

BASE_DIR = Path(__file__).resolve().parent.parent  # this gives you the project root (SeeFoodServer/)
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

# Add ../SeeFoodModel to the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "SeeFoodModel"))

from model import predict, configure_model

class DeployedPredictor():
    def __init__(self, img_size= (512, 512), model_path = ARTIFACT_DIR/"model_state.pth", model_config_path = ARTIFACT_DIR/"model_config.json", dict_path = ARTIFACT_DIR/"idx_to_label.json"):
        self.img_size = (512, 512)
        self.to_tensor = ToTensor()
        # load model and save in object
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        model = configure_model(model_config)
        model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
        self.model = model
        # load dict and save in object
        with open(dict_path, 'r') as f:
            idx_to_label = json.load(f)
        self.idx_to_label = idx_to_label 
        
    def predict_image(self, base64_img): 
        decoded_img = self.decode_base64_image(base64_img)
        img = self.resize_image(decoded_img, size = self.img_size)
        x = self.to_tensor(img)
        with torch.no_grad():
            output = predict(self.model, x.unsqueeze(0))
            idx = output[:, 0][torch.argmax(output[:, 1])]
            return self.idx_to_label[f'{int(idx)}']
    
    def resize_image(self, image, size=(512, 512)):
        return image.resize(size)
        
    def decode_base64_image(self, b64_string):
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image