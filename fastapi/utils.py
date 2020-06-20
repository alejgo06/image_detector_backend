import os
import numpy as np
import math
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt


def predict_new_image(imgoriginal, model, names, threshold=0.7,device='cpu'):
    im = imgoriginal.copy()
    im = im / 255
    im = torch.tensor(im.transpose(2, 0, 1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model([im.to(device)])

    boxes = prediction[0]['boxes'].to('cpu').numpy().tolist()
    scores = prediction[0]['scores'].to('cpu').numpy().tolist()
    labels = prediction[0]['labels'].to('cpu').numpy().tolist()

    jsonOutput={}
    for i in range(len(boxes)):
        coordenadas = boxes[i]
        confianza = scores[i]
        if confianza > threshold:
            jsonOutput.update({str(i):{'coordinates':coordenadas,
                                       'confidence':confianza,
                                       'label':names[labels[i]]}
                               })

    return jsonOutput


