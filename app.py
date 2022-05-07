import json
import re
import flask
import io
import string
import time
import os
import numpy as np
##import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify
import numpy as np

transform=transforms.Resize(224)
##model = tf.keras.models.load_model('resnet50_food_model')
class CustomEfficientNet(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 2)

    def forward(self, x):
        x = self.model(x)
        return x

def prepare_image(image_bytes):
    """""
    Transform image into required DenseNet format: 224x224 with 3 RGB channels and normalized.
    Return the corresponding tensor.
    """
    my_transforms = A.Compose([
            A.Resize(512, 512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    # ret,bw = cv2.threshold(image_bytes, 512, 512, cv2.THRESH_BINARY)
    image = Image.open(io.BytesIO(image_bytes))
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return my_transforms(image=np.array(image_np))['image'].unsqueeze(0)

model = CustomEfficientNet('efficientnet_b4')
state_dict = torch.load('model_b4.pth', map_location=torch.device('cpu'))['model']
model.load_state_dict(state_dict)
model.eval()

def predict_result(image_bytes):
    """For given image bytes, predict the label using the pretrained DenseNet
    """
    tensor = prepare_image(image_bytes)
    ##prepare_image(image_bytes)
    outputs = model.forward(tensor)
    proba, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    #class_name, human_label = imagenet_mapping[predicted_idx]
    return predicted_idx

##ef prepare_image(img):

##def predict_result(img):


app = Flask(__name__)
ALLOWED_EXTENSIONS= {'png','jpg','jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def infer_image():
    # if 'file' not in request.files:
    #     return "Please try again. The Image doesn't exist"
    
    # file = request.files.get('file')

    # if not file:
    #     return

    # img_bytes = file.read()
    # # img = prepare_image(img_bytes)

    # return jsonify(prediction=predict_result(img_bytes))
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error':'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            prediction = predict_result(img_bytes)
            data = {'prediction': prediction}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})

    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')