import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pandas as pd
from torch.utils.data import Dataset
from utils import get_mask
import torchvision.transforms as transforms
import transforms as T

print("v1")

def get_gatitos_df(path_to_json):
    with open(path_to_json) as json_file:
        data = json.load(json_file)

    lista_anotaciones_gatos = []
    for i in range(len(data['annotations'])):
        if data['annotations'][i]['category_id'] == 17:
            if data['annotations'][i]['image_id'] not in [259557, 554737]:
                lista_anotaciones_gatos.append(data['annotations'][i])
    df_anotacion_gatitos = pd.DataFrame(lista_anotaciones_gatos)
    dfImages = pd.DataFrame(data['images'])
    df = pd.merge(df_anotacion_gatitos, dfImages, how='left', left_on=['image_id'], right_on=['id'])
    return df

def get_df_from_coco(path_to_json, categories=[17]):
    with open(path_to_json) as json_file:
        data = json.load(json_file)

    annotations_list = []
    for i in range(len(data['annotations'])):
        if data['annotations'][i]['category_id'] in categories:
            if (data['annotations'][i]['bbox'][0]==0) and (data['annotations'][i]['bbox'][1]==0):
                print(i)
            elif (data['annotations'][i]['bbox'][2]==0) or (data['annotations'][i]['bbox'][3]==0) or \
                    (data['annotations'][i]['bbox'][0]==data['annotations'][i]['bbox'][2]) or \
                (data['annotations'][i]['bbox'][1]==data['annotations'][i]['bbox'][3]):
                print(i)
            else:
                if len(data['annotations'][i]['bbox'])<4:
                    print(f"hay menos de 4 boxes en {i}")
                else:
                    box=data['annotations'][i]['bbox']
                    xmin=box[0]
                    xmax=box[0]+box[2]
                    ymin=box[1]
                    ymax=box[1]+box[3]
                    area = (xmax - xmin) * (ymax - ymin)
                    if (area<10) | (data['annotations'][i]['area']<10):
                        print(f"area por debajo en de 10 pixeles en {i}")
                    else:
                        annotations_list.append(data['annotations'][i])

    df_annotations = pd.DataFrame(annotations_list).sample(n=6000, random_state=10)# muestreo
    # 30k unas 2 horas
    # 15k unas 1 hora
    dfImages = pd.DataFrame(data['images'])
    df = pd.merge(df_annotations, dfImages, how='left', left_on=['image_id'], right_on=['id'])
    return df


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


# load all image files, sorting them to ensure that they are aligned
#self.df = get_gatitos_df(path_json)#get_df_from_coco(path_json, categories)
#mask = get_mask(coordenates=self.df['bbox'][idx], height=self.df['height'][idx],
                        #width=self.df['width'][idx])



class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, path_json, path_images,categories, transforms=None):
        self.path_json = path_json
        self.path_images = path_images
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.df = get_df_from_coco(path_json, categories)
        self.imgs = self.df['file_name']

    def __getitem__(self, idx):
        # load images ad masks

        img_path = os.path.join(self.path_images, self.df['file_name'][idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = get_mask(coordenates=self.df['bbox'][idx], height=self.df['height'][idx],
                        width=self.df['width'][idx])

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # print(boxes)
        # print(idx)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])



        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class GatitosDataset(torch.utils.data.Dataset):
    def __init__(self, path_json, path_images, transforms=None):
        self.path_json = path_json
        self.path_images = path_images
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.df_gatitos = get_gatitos_df(path_json)
        self.imgs = self.df_gatitos['file_name']

    def __getitem__(self, idx):
        # load images ad masks

        img_path = os.path.join(self.path_images, self.df_gatitos['file_name'][idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = get_mask(coordenates=self.df_gatitos['bbox'][idx], height=self.df_gatitos['height'][idx],
                        width=self.df_gatitos['width'][idx])

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # print(boxes)
        # print(idx)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # self.df_gatitos['area'][idx]#
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)