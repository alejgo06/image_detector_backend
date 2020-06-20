from fastapi import FastAPI
from typing import List
from fastapi import FastAPI, UploadFile, File
import numpy as np
from starlette.requests import Request
import io
from PIL import Image
import base64
import cv2
app = FastAPI()


from models.models import get_instance_segmentation_model
from utils import predict_new_image
import torch
import os

model = get_instance_segmentation_model(3)
model.load_state_dict(torch.load(os.getcwd()+"/models/generator_14.pth",map_location=torch.device('cpu')))
model.to('cpu')
model.eval()

@app.post("/predict")
async def analyse(image_file_read: bytes = File(...)):
    file = base64.b64encode(image_file_read)
    jpg_original = base64.b64decode(file)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    original_image = cv2.imdecode(jpg_as_np, flags=1)
    preidction=predict_new_image(original_image, model, names=['cat', 'dog'], threshold=0.7,device='cpu')
    return preidction
