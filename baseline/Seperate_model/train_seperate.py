import multiprocessing
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from dataset_seperate import *
from hyperparameter_seperate import HyperParameter
from loss import *
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


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

model_mask=torchvision.models.resnet50(pretrained=True)
model_mask.fc=nn.Linear(model_mask.fc.in_features,HyperParameter.NUM_MASK_CLASS)
model_mask.to(device)

##################################################################
model_age=torchvision.models.resnet50(pretrained=True)
model_age.fc=nn.Linear(model_age.fc.in_features,HyperParameter.NUM_AGE_CLASS)
model_age.to(device)

model_gender=torchvision.models.resnet50(pretrained=True)
model_gender.fc=nn.Linear(model_gender.fc.in_features,HyperParameter.NUM_GENDER_CLASS)
model_gender.to(device)
##################################################################
transform=transforms.Compose([
    transforms.Resize(HyperParameter.RESIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])


train_idx,val_idx=SplitByHumanDataset.split_train_val()
train_dataset=SplitByHumanDataset(train_idx, transform)
val_dataset=SplitByHumanDataset(val_idx, transform)

criterion=FocalLoss()
optimizer_mask=torch.optim.Adam(params=model_mask.parameters(),lr=HyperParameter.LEARNING_RATE)
optimizer_age=torch.optim.Adam(params=model_age.parameters(),lr=HyperParameter.LEARNING_RATE)
optimizer_gender=torch.optim.Adam(params=model_gender.parameters(),lr=HyperParameter.LEARNING_RATE)

train_loader = DataLoader(
    train_dataset,
    batch_size=HyperParameter.BATCH_SIZE,
    num_workers=multiprocessing.cpu_count() // 2,
    shuffle=True,
    pin_memory=use_cuda,
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
#########################################################
best_val_acc_mask = 0
best_val_acc_age = 0
best_val_acc_gender = 0

best_val_loss_mask = float('inf')
best_val_loss_age = float('inf')
best_val_loss_gender = float('inf')

best_val_f1_mask=0
best_val_f1_age=0
best_val_f1_gender=0
#########################################################
for epoch in range(HyperParameter.EPOCH):
    # train loop
    model_mask.train()
    model_age.train()
    model_gender.train()
    
    loss_value_mask = 0
    loss_value_age = 0
    loss_value_gender = 0
    
    matches_mask = 0
    matches_age = 0 
    matches_gender = 0
    
    label_list_mask=[]
    label_list_age=[]
    label_list_gender=[]
    
    pred_list_mask=[]
    pred_list_age=[]
    pred_list_gender=[]
    for idx, train_batch in enumerate(train_loader):
        inputs, labels = train_batch
        inputs = inputs.to(device)
        
        labels_mask = SplitByHumanDataset.mask(*labels).to(device)
        labels_age = SplitByHumanDataset.age(*labels).to(device)
        labels_gender = SplitByHumanDataset.gender(*labels).to(device)
        
        optimizer_mask.zero_grad()
        optimizer_age.zero_grad()
        optimizer_gender.zero_grad()

        outs_mask = model_mask(inputs)
        outs_age = model_age(inputs)
        outs_gender = model_gender(inputs)
        
        preds_mask = torch.argmax(outs_mask, dim=-1)
        preds_age = torch.argmax(outs_age, dim=-1)
        preds_gender = torch.argmax(outs_gender, dim=-1)
        
        loss_mask = criterion(outs_mask, labels_mask)
        loss_age = criterion(outs_age, labels_age)
        loss_gender = criterion(outs_gender, labels_gender)

        label_list_mask +=labels_mask.detach().cpu()
        label_list_age +=labels_age.detach().cpu()
        label_list_gender +=labels_gender.detach().cpu()
        
        pred_list_mask +=preds_mask.detach().cpu()
        pred_list_age +=preds_age.detach().cpu()
        pred_list_gender +=preds_gender.detach().cpu()

        loss_mask.backward()
        loss_age.backward()
        loss_gender.backward()
        
        optimizer_mask.step()
        optimizer_age.step()
        optimizer_gender.step()

        loss_value_mask += loss_mask.item()
        loss_value_age += loss_age.item()
        loss_value_gender += loss_gender.item()
        
        matches_mask += (preds_mask == labels_mask).sum().item()
        matches_age += (preds_age == labels_age).sum().item()
        matches_gender += (preds_gender == labels_gender).sum().item()
        
        if (idx + 1) % HyperParameter.LOG_INTERVAL == 0:
            train_loss_mask = loss_value_mask / HyperParameter.LOG_INTERVAL
            train_loss_age = loss_value_age / HyperParameter.LOG_INTERVAL
            train_loss_gender = loss_value_gender / HyperParameter.LOG_INTERVAL
            
            train_acc_mask = matches_mask / HyperParameter.BATCH_SIZE / HyperParameter.LOG_INTERVAL
            train_acc_age = matches_age / HyperParameter.BATCH_SIZE / HyperParameter.LOG_INTERVAL
            train_acc_gender = matches_gender / HyperParameter.BATCH_SIZE / HyperParameter.LOG_INTERVAL
            
            train_f1_mask=f1_score(label_list_mask,pred_list_mask,average="macro")
            train_f1_age=f1_score(label_list_age,pred_list_age,average="macro")
            train_f1_gender=f1_score(label_list_gender,pred_list_gender,average="macro")
            print(
                f"Epoch[{epoch}/{HyperParameter.EPOCH}]({idx + 1}/{len(train_loader)}) || "
                f"training mask F1 Score {train_f1_mask:4.4} || training mask loss {train_loss_mask:4.4} || training mask accuracy {train_acc_mask:4.2%} ||"
                f"training age F1 Score {train_f1_age:4.4} || training age loss {train_loss_age:4.4} || training age accuracy {train_acc_age:4.2%} ||"
                f"training gender F1 Score {train_f1_gender:4.4} || training gender loss {train_loss_gender:4.4} || training gender accuracy {train_acc_gender:4.2%} ||"
            )
            logger.add_scalar("Train/Loss_mask", train_loss_mask, epoch * len(train_loader) + idx)
            logger.add_scalar("Train/Loss_age", train_loss_age, epoch * len(train_loader) + idx)
            logger.add_scalar("Train/Loss_gender", train_loss_gender, epoch * len(train_loader) + idx)
            
            logger.add_scalar("Train/Accuracy_mask", train_acc_mask, epoch * len(train_loader) + idx)
            logger.add_scalar("Train/Accuracy_age", train_acc_age, epoch * len(train_loader) + idx)
            logger.add_scalar("Train/Accuracy_gender", train_acc_gender, epoch * len(train_loader) + idx)
            
            logger.add_scalar("Train/F1_Score_mask", train_acc_mask, epoch * len(train_loader) + idx)
            logger.add_scalar("Train/F1_Score_age", train_acc_age, epoch * len(train_loader) + idx)
            logger.add_scalar("Train/F1_Score_gender", train_acc_gender, epoch * len(train_loader) + idx)
            loss_value_mask = 0
            loss_value_gender = 0
            loss_value_age = 0
            matches_mask = 0
            matches_gender = 0
            matches_age = 0
            label_list_mask=[]
            label_list_gender=[]
            label_list_age=[]
            pred_list_mask=[]
            pred_list_gender=[]
            pred_list_age=[]
            
    # evaluation session
    with torch.no_grad():
        model_mask.eval()
        model_age.eval()
        model_gender.eval()
        
        val_loss_mask=0
        val_loss_age=0
        val_loss_gender=0
        
        val_acc_mask=0
        val_acc_age=0
        val_acc_gender=0
        label_list_mask=[]
        label_list_age=[]
        label_list_gender=[]
        
        pred_list_mask=[]
        pred_list_age=[]
        pred_list_gender=[]
        for val_batch in val_loader:
            inputs, labels = val_batch
            inputs = inputs.to(device)

            labels_mask = SplitByHumanDataset.mask(*labels).to(device)
            labels_age = SplitByHumanDataset.age(*labels).to(device)
            labels_gender = SplitByHumanDataset.gender(*labels).to(device)
            
            outs_mask = model_mask(inputs)
            outs_age = model_age(inputs)
            outs_gender = model_gender(inputs)
            
            preds_mask = torch.argmax(outs_mask, dim=-1)
            preds_age = torch.argmax(outs_age, dim=-1)
            preds_gender = torch.argmax(outs_gender, dim=-1)
            
            label_list_mask+=labels_mask.detach().cpu()
            label_list_age+=labels_age.detach().cpu()
            label_list_gender+=labels_gender.detach().cpu()
            
            pred_list_mask+=preds_mask.detach().cpu()
            pred_list_age+=preds_age.detach().cpu()
            pred_list_gender+=preds_gender.detach().cpu()
            
            val_acc_mask+=(labels_mask==preds_mask).sum().item()
            val_acc_age+=(labels_age==preds_age).sum().item()
            val_acc_gender+=(labels_gender==preds_gender).sum().item()
            
            val_loss_mask+= criterion(outs_mask, labels_mask).item()
            val_loss_age+= criterion(outs_age, labels_age).item()
            val_loss_gender+= criterion(outs_gender, labels_gender).item()


        val_loss_mask = val_loss_mask / len(val_loader)
        val_loss_age = val_loss_age / len(val_loader)
        val_loss_gender = val_loss_gender / len(val_loader)
        
        val_acc_mask = val_acc_mask / len(label_list_mask)
        val_acc_age = val_acc_age / len(label_list_age)
        val_acc_gender = val_acc_gender / len(label_list_gender)
        
        val_f1_mask= f1_score(label_list_mask,pred_list_mask,average="macro")
        val_f1_age= f1_score(label_list_age,pred_list_age,average="macro")
        val_f1_gender= f1_score(label_list_gender,pred_list_gender,average="macro")
        
        best_val_loss_mask = min(best_val_loss_mask, val_loss_mask)
        best_val_loss_age = min(best_val_loss_age, val_loss_age)
        best_val_loss_gender = min(best_val_loss_gender, val_loss_gender)
        
        best_val_acc_mask = max(best_val_acc_mask, val_acc_mask)
        best_val_acc_age = max(best_val_acc_age, val_acc_age)
        best_val_acc_gender = max(best_val_acc_gender, val_acc_gender)
        if val_f1_mask > best_val_f1_mask:
            print(f"New best model_mask for f1 score : {val_f1_mask:4.2%}! saving the best model..")
            torch.save(model_mask.state_dict(), f"{HyperParameter.SAVE_DIR}/best_model_mask.pth")
            best_val_f1_mask = val_f1_mask
        if val_f1_age > best_val_f1_age:
            print(f"New best model_age for f1 score : {val_f1_age:4.2%}! saving the best model..")
            torch.save(model_age.state_dict(), f"{HyperParameter.SAVE_DIR}/best_model_age.pth")
            best_val_f1_age = val_f1_age
        if val_f1_gender > best_val_f1_gender:
            print(f"New best model_gender for f1 score : {val_f1_gender:4.2%}! saving the best model..")
            torch.save(model_gender.state_dict(), f"{HyperParameter.SAVE_DIR}/best_model_gender.pth")
            best_val_f1_gender = val_f1_gender
            
        torch.save(model_mask.state_dict(), f"{HyperParameter.SAVE_DIR}/last_model_mask.pth")
        torch.save(model_age.state_dict(), f"{HyperParameter.SAVE_DIR}/last_model_age.pth")
        torch.save(model_gender.state_dict(), f"{HyperParameter.SAVE_DIR}/last_model_gender.pth")
        print(f"[Val mask] f1 : {val_f1_mask:4.2%}, acc : {val_acc_mask:4.2%}, loss: {val_loss_mask:4.2} || best f1 : {best_val_f1_mask:4.2%}, best acc : {best_val_acc_mask:4.2%}, best loss: {best_val_loss_mask:4.2}")
        print(f"[Val age] f1 : {val_f1_age:4.2%}, acc : {val_acc_age:4.2%}, loss: {val_loss_age:4.2} || best f1 : {best_val_f1_age:4.2%}, best acc : {best_val_acc_age:4.2%}, best loss: {best_val_loss_age:4.2}")
        print(f"[Val gender] f1 : {val_f1_gender:4.2%}, acc : {val_acc_gender:4.2%}, loss: {val_loss_gender:4.2} || best f1 : {best_val_f1_gender:4.2%}, best acc : {best_val_acc_gender:4.2%}, best loss: {best_val_loss_gender:4.2}")
        logger.add_scalar("Val/Loss_mask", val_loss_mask, epoch)
        logger.add_scalar("Val/Loss_age", val_loss_age, epoch)
        logger.add_scalar("Val/Loss_gender", val_loss_gender, epoch)
        
        logger.add_scalar("Val/Accuracy_mask", val_acc_mask, epoch)
        logger.add_scalar("Val/Accuracy_age", val_acc_age, epoch)
        logger.add_scalar("Val/Accuracy_gender", val_acc_gender, epoch)
        
        logger.add_scalar("Val/F1_Score_mask",val_f1_mask,epoch)
        logger.add_scalar("Val/F1_Score_age",val_f1_age,epoch)
        logger.add_scalar("Val/F1_Score_gender",val_f1_gender,epoch)
        print("-"*60)
