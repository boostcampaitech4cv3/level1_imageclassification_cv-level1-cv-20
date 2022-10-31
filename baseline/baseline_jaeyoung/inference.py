import multiprocessing
import os

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from albumentations.pytorch import ToTensorV2
from dataset import *
from hyperparameter import HyperParameter
from torch.utils.data import DataLoader

from model import Deit3Base224, ResNet18, ResnextModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model=ResNet18(HyperParameter.NUM_CLASS).cuda()

model.load_state_dict(torch.load(os.path.join(HyperParameter.SAVE_DIR,"best_model.pth"),map_location=device))
model.eval()

transform=A.Compose([
    # A.Resize(*HyperParameter.RESIZE),
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
preds = []
with torch.no_grad():
    for idx, images in enumerate(test_loader):
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        preds.extend(pred.cpu().numpy())
test_df['ans'] = preds
save_path = os.path.join(HyperParameter.SAVE_DIR,'output.csv')
test_df.to_csv(save_path, index=False)
print(f"Inference Done! Inference result saved at {save_path}")