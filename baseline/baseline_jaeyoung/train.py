import multiprocessing
import random

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from albumentations.pytorch import ToTensorV2
from dataset import *
from hyperparameter import HyperParameter
from loss import *
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler

from model import (ConvnextModel, Deit3Base224,EfficientNetB0, EfficientNetB4, ResNet18,
                   ResnextModel,SwinTransformer)


def seed_config(seed):
    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
seed_config(HyperParameter.SEED)

model = SwinTransformer(HyperParameter.NUM_CLASS).cuda()


transform_train=A.Compose([
    A.CenterCrop(320,256),
    A.Resize(224,224,interpolation=cv2.INTER_CUBIC),
    A.Normalize(),
    A.HorizontalFlip(),
    ToTensorV2(),
])

transform_val=A.Compose([
    A.CenterCrop(320,256),
    A.Resize(224,224,interpolation=cv2.INTER_CUBIC),
    A.Normalize(),
    ToTensorV2(),
])



train_idx,val_idx=SplitByHumanDataset.split_train_val()
train_dataset=SplitByHumanDataset(train_idx, transform_train,train=True)
val_dataset=SplitByHumanDataset(val_idx, transform_val,train=False)

criterion=FocalLoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=HyperParameter.LEARNING_RATE,weight_decay=1e-8)
scheduler=StepLR(optimizer, 5, gamma=0.5)

class Sampler(ImbalancedDatasetSampler):
    def _get_labels(self, dataset):
        return dataset.classes

train_loader = DataLoader(
    train_dataset,
    batch_size=HyperParameter.BATCH_SIZE,
    num_workers=multiprocessing.cpu_count() // 2,
    pin_memory=use_cuda,
    sampler=Sampler(train_dataset),
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=HyperParameter.BATCH_SIZE,
    num_workers=multiprocessing.cpu_count() // 2,
    shuffle=False,
    pin_memory=use_cuda,
    drop_last=True,
)
logger = SummaryWriter(log_dir=HyperParameter.SAVE_DIR)

best_val_acc = 0
best_val_loss = float('inf')
best_val_f1=0
for epoch in range(HyperParameter.EPOCH):
    # train loop
    model.train()
    loss_value = 0
    matches = 0
    label_list=[]
    pred_list=[]
    for idx, train_batch in enumerate(train_loader):
        inputs, labels = train_batch
        inputs = inputs.to(device)
        labels = SplitByHumanDataset.multi_to_single(*labels).to(device)
        # labels = labels[2].to(device)

        optimizer.zero_grad()

        outs = model(inputs)
        preds = torch.argmax(outs, dim=-1)
        loss = criterion(outs, labels)

        label_list+=labels.detach().cpu()
        pred_list+=preds.detach().cpu()

        loss.backward()
        optimizer.step()

        loss_value += loss.item()
        matches += (preds == labels).sum().item()
        if (idx + 1) % HyperParameter.LOG_INTERVAL == 0:
            train_loss = loss_value / HyperParameter.LOG_INTERVAL
            train_acc = matches / HyperParameter.BATCH_SIZE / HyperParameter.LOG_INTERVAL
            train_f1=f1_score(label_list,pred_list,average="macro")
            print(
                f"Epoch[{epoch}/{HyperParameter.EPOCH}]({idx + 1}/{len(train_loader)}) || "
                f"training F1 Score {train_f1:4.4} || training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
            )
            logger.add_scalar("Train/Loss", train_loss, epoch * len(train_loader) + idx)
            logger.add_scalar("Train/Accuracy", train_acc, epoch * len(train_loader) + idx)
            logger.add_scalar("Train/F1_Score", train_acc, epoch * len(train_loader) + idx)
            loss_value = 0
            matches = 0
            label_list=[]
            pred_list=[]
    # evaluation session
    with torch.no_grad():
        model.eval()
        val_loss=0
        val_acc=0
        label_list=[]
        pred_list=[]
        for val_batch in val_loader:
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = SplitByHumanDataset.multi_to_single(*labels).to(device)
            # labels = labels[2].to(device)
            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            label_list+=labels.detach().cpu()
            pred_list+=preds.detach().cpu()
            val_acc+=(labels==preds).sum().item()
            val_loss+= criterion(outs, labels).item()


        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(label_list)
        val_f1= f1_score(label_list,pred_list,average="macro")
        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)
        if val_f1 > best_val_f1:
            print(f"New best model for f1 score : {val_f1:4.2%}! saving the best model..")
            torch.save(model.state_dict(), f"{HyperParameter.SAVE_DIR}/best_model.pth")
            best_val_f1 = val_f1
        torch.save(model.state_dict(), f"{HyperParameter.SAVE_DIR}/last_model.pth")
        print(f"[Val] f1 : {val_f1:4.2%}, acc : {val_acc:4.2%}, loss: {val_loss:4.2} || best f1 : {best_val_f1:4.2%}, best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}")
        logger.add_scalar("Val/Loss", val_loss, epoch)
        logger.add_scalar("Val/Accuracy", val_acc, epoch)
        logger.add_scalar("Val/F1_Score",val_f1,epoch)
        print("-"*60)