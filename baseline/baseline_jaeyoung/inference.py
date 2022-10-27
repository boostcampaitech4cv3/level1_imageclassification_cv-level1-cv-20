import multiprocessing
import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from dataset import *
from hyperparameter import HyperParameter
from torch.utils.data import DataLoader
from torchvision import transforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model=torchvision.models.resnet50(pretrained=True)
model.fc=nn.Linear(model.fc.in_features,HyperParameter.NUM_CLASS)
model.to(device)

model.load_state_dict(torch.load(os.path.join(HyperParameter.SAVE_DIR,"best_model.pth"),map_location=device))
model.eval()

transform=transforms.Compose([
    transforms.Resize(HyperParameter.RESIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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