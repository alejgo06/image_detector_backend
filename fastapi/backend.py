from fastapi import FastAPI
from typing import List
from fastapi import FastAPI, UploadFile, File
import numpy as np
from starlette.requests import Request
import io
from PIL import Image
import base64
import cv2
app = FastAPI(title="Image detector",
    description="Api",
    version="21.02.22.18.11")


from models.models import get_instance_segmentation_model
from utils import predict_new_image
import torch
import os



@app.post("/predict")
async def analyse(image_file_read: bytes = File(...)):
    file = base64.b64encode(image_file_read)
    jpg_original = base64.b64decode(file)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    original_image = cv2.imdecode(jpg_as_np, flags=1)
    model = get_instance_segmentation_model(3)
    model.load_state_dict(torch.load(os.getcwd() + "/models/generator_14.pth", map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    preidction=predict_new_image(original_image, model, names=['cat', 'dog'], threshold=0.7,device='cpu')
    return preidction

@app.post("/predictPEOPLE")
async def analysePEOPLE(image_file_read: bytes = File(...)):
    file = base64.b64encode(image_file_read)
    jpg_original = base64.b64decode(file)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    original_image = cv2.imdecode(jpg_as_np, flags=1)
    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load(os.getcwd() + "/models/generator_13PEOPLE.pth", map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    preidction=predict_new_image(original_image, model, names=['background','person'], threshold=0.7,device='cpu')
    return preidction

@app.post("/predictHORSES")
async def analyseHORSES(image_file_read: bytes = File(...)):
    file = base64.b64encode(image_file_read)
    jpg_original = base64.b64decode(file)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    original_image = cv2.imdecode(jpg_as_np, flags=1)
    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load(os.getcwd() + "/models/generator_4HORSES.pth", map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    preidction=predict_new_image(original_image, model, names=[ 'background','horse'], threshold=0.7,device='cpu')
    return preidction

@app.post("/predictCATS")
async def analyseCATS(image_file_read: bytes = File(...)):
    file = base64.b64encode(image_file_read)
    jpg_original = base64.b64decode(file)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    original_image = cv2.imdecode(jpg_as_np, flags=1)
    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load(os.getcwd() + "/models/generator_18CATS.pth", map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    preidction=predict_new_image(original_image, model, names=[ 'background','cat'], threshold=0.7,device='cpu')
    return preidction

@app.post("/predictDOGS")
async def analyseDOGS(image_file_read: bytes = File(...)):
    file = base64.b64encode(image_file_read)
    jpg_original = base64.b64decode(file)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    original_image = cv2.imdecode(jpg_as_np, flags=1)
    model = get_instance_segmentation_model(2)
    model.load_state_dict(torch.load(os.getcwd() + "/models/generator_21DOGS.pth", map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    preidction=predict_new_image(original_image, model, names=['background','dog'], threshold=0.7,device='cpu')
    return preidction
