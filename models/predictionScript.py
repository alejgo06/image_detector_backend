import os
import numpy as np
import math
import itertools
import sys
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import json
import pandas as pd
from PIL import Image
import sys
from pathlib import Path
import modelutils
from models import get_instance_segmentation_model,train_one_epoch, _get_iou_types
from modelutils import precision_recall, get_precisionDF
import tqdm

device='cpu'


import cv2
import matplotlib.pyplot as plt

def predict(img):
    model = get_instance_segmentation_model(3)
    model.load_state_dict(torch.load("generator_14.pth",map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load("saved_models_cat_dogs_horse/generator_9.pth"))
    model.to(device)
    model.eval()
    im=img/255
    im=torch.tensor(im.transpose(2, 0, 1),dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model([im.to(device)])
    return prediction


def predict_new_image(path_to_file,names,threshold=0.7):
    imgoriginal=cv2.imread(path_to_file)
    plt.imshow(cv2.cvtColor(imgoriginal, cv2.COLOR_BGR2RGB))
    plt.show()
    im=imgoriginal.copy()
    prediction=predict(im)
    
    boxes=prediction[0]['boxes'].to('cpu').numpy().tolist()
    scores=prediction[0]['scores'].to('cpu').numpy().tolist()
    labels=prediction[0]['labels'].to('cpu').numpy().tolist()
    
    for i in range(len(boxes)):
        copyimage=imgoriginal.copy()
        coordenadas=boxes[i]
        confianza=scores[i]
        if confianza>threshold:
            print(names[labels[i]])
            cv2.rectangle(copyimage,(int(coordenadas[0]),
                           int(coordenadas[1])),
                          (int(coordenadas[2]),
                           int(coordenadas[3])),(0,255,0),6)
        plt.imshow(cv2.cvtColor(copyimage, cv2.COLOR_BGR2RGB))
        plt.show()