import multiprocessing
import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from dataset_seperate import *
from hyperparameter_seperate import HyperParameter
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
###################################################################
model_mask=torchvision.models.resnet18(pretrained=True)
model_mask.fc=nn.Linear(model_mask.fc.in_features,HyperParameter.NUM_MASK_CLASS)
model_mask.to(device)

model_age=torchvision.models.resnet18(pretrained=True)
model_age.fc=nn.Linear(model_age.fc.in_features,HyperParameter.NUM_AGE_CLASS)
model_age.to(device)

model_gender=torchvision.models.resnet18(pretrained=True)
model_gender.fc=nn.Linear(model_gender.fc.in_features,HyperParameter.NUM_GENDER_CLASS)
model_gender.to(device)

model_mask.load_state_dict(torch.load(os.path.join(HyperParameter.SAVE_DIR,"best_model_mask.pth"),map_location=device))
model_mask.eval()

model_age.load_state_dict(torch.load(os.path.join(HyperParameter.SAVE_DIR,"best_model_age.pth"),map_location=device))
model_age.eval()

model_gender.load_state_dict(torch.load(os.path.join(HyperParameter.SAVE_DIR,"best_model_gender.pth"),map_location=device))
model_gender.eval()
###################################################################

transform=A.Compose([
            #A.CenterCrop(320,256),
            A.Resize(HyperParameter.RESIZE[0],HyperParameter.RESIZE[1],interpolation=cv2.INTER_CUBIC),
            A.CLAHE(p=1),
            A.Normalize(),
            ToTensorV2(),
        ])

test_df=pd.read_csv(HyperParameter.TEST_CSV_DIR)
test_dataset=TestDataset(df=test_df,transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=HyperParameter.BATCH_SIZE,
    num_workers=multiprocessing.cpu_count() // 2,
    shuffle=False,
    pin_memory=use_cuda,
    drop_last=False,
)
preds_mask = []
preds_age = []
preds_gender = []
with torch.no_grad():
    for idx, images in enumerate(test_loader):
        images = images.to(device)
        pred_mask = model_mask(images)
        pred_mask = pred_mask.argmax(dim=-1)
        preds_mask.extend(pred_mask.cpu().numpy())
        
        pred_age = model_age(images)
        pred_age = pred_age.argmax(dim=-1)
        preds_age.extend(pred_age.cpu().numpy())
        
        pred_gender = model_gender(images)
        pred_gender = pred_gender.argmax(dim=-1)
        preds_gender.extend(pred_gender.cpu().numpy())


test_df['ans'] = [(6*x + y + 3*z) for x,y,z in zip(preds_mask, preds_age, preds_gender)]
save_path = os.path.join(HyperParameter.SAVE_DIR,'output.csv')
test_df.to_csv(save_path, index=False)
print(f"Inference Done! Inference result saved at {save_path}")