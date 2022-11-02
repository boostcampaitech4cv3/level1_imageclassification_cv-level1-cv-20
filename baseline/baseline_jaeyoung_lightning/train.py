import io
import multiprocessing
import random
import sys

import albumentations as A
import cv2
import loss_functions
import matplotlib.pyplot as plt
import models
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from albumentations.pytorch import ToTensorV2
from datasets import *
from hyperparameters import HyperParameter
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset


class MaskModel(pl.LightningModule):
    def preprocessing(self):
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
            A.Normalize(mean=(0.548, 0.504, 0.479),std=(0.237, 0.247, 0.246)),
            ToTensorV2(),
        ])

        train_idx,val_idx=SplitByHumanDataset.split_train_val()
        train_dataset=SplitByHumanDataset(train_idx, transform_train,train=True)
        val_dataset=SplitByHumanDataset(val_idx, transform_val,train=False)


        self.train_loader = DataLoader(
            train_dataset,
            batch_size=HyperParameter.BATCH_SIZE,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=HyperParameter.BATCH_SIZE,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

    def log_confusion_matrix(self,metric):
        x=metric.compute().cpu().numpy()
        fig = plt.figure(figsize = (20,10),dpi=100)
        sns.heatmap(x,annot=True,cmap="Blues",fmt='g')
        return fig
        
    def __init__(self, model_name,num_class,learning_rate,loss_funtion_name):
        super().__init__()
        self.model=getattr(models,model_name)(num_class)
        self.lr=learning_rate
        self.criterion=getattr(loss_functions,loss_funtion_name)()
        # setup metrics module
        self.train_acc= torchmetrics.Accuracy()
        self.val_acc=torchmetrics.Accuracy()
        self.train_loss= torchmetrics.MeanMetric()
        self.val_loss=torchmetrics.MeanMetric()
        self.train_f1= torchmetrics.F1Score(num_class,average="macro")
        self.val_f1=torchmetrics.F1Score(num_class,average="macro")
        self.num_class=num_class
        self.train_conf_matrix=torchmetrics.ConfusionMatrix(num_classes=num_class)
        self.val_conf_matrix=torchmetrics.ConfusionMatrix(num_classes=num_class)
        self.save_hyperparameters()
        self.preprocessing()

    def training_step(self, batch, batch_idx):
        inputs,labels = batch
        labels = SplitByHumanDataset.multi_to_single(*labels)
        outs=self.model(inputs)
        preds = torch.argmax(outs, dim=-1)
        loss= self.criterion(outs,labels)
        self.train_loss(loss)
        self.train_acc(preds,labels)
        self.train_f1(preds,labels)
        self.train_conf_matrix(preds,labels)
        self.log("Train/loss", self.train_loss,on_step=True,on_epoch=True)
        self.log("Train/acc",self.train_acc,on_step=True,on_epoch=True)
        self.log("Train/f1",self.train_f1,on_step=True,on_epoch=True)
        return loss
    
    def training_epoch_end(self, outputs) -> None:
        tensorboard=self.logger.experiment
        tensorboard.add_figure("Train/confusion",self.log_confusion_matrix(self.train_conf_matrix),self.current_epoch)
        self.train_conf_matrix.reset()
        
    
    def validation_step(self, batch, batch_idx):
        inputs,labels = batch
        labels = SplitByHumanDataset.multi_to_single(*labels)
        outs=self.model(inputs)
        preds = torch.argmax(outs, dim=-1)
        loss= self.criterion(outs,labels)
        self.val_loss(loss)
        self.val_acc(preds,labels)
        self.val_f1(preds,labels)
        self.val_conf_matrix(preds,labels)
        self.log("Validation/loss", self.val_loss,on_step=False,on_epoch=True)
        self.log("Validation/acc",self.val_acc,on_step=False,on_epoch=True)
        self.log("Validation/f1",self.val_f1,on_step=False,on_epoch=True)
    
    def validation_epoch_end(self, outputs) -> None:
        tensorboard=self.logger.experiment
        tensorboard.add_figure("Validation/confusion",self.log_confusion_matrix(self.val_conf_matrix),self.current_epoch)
        self.val_conf_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

pl.seed_everything(HyperParameter.SEED,workers=True)
model=MaskModel(
    model_name="EfficientNetB0",
    num_class=18,
    learning_rate=HyperParameter.LEARNING_RATE,
    loss_funtion_name="FocalLoss"
)



checkpoint1=ModelCheckpoint(
    monitor="Validation/f1",
    filename='epoch{epoch:02d}-val_f1={Validation/f1:.3f}-val_acc={Validation/acc:.3f}',
    save_top_k=3,
    mode="max",
    save_on_train_epoch_end=False,
    auto_insert_metric_name=False
     
)

checkpoint2=ModelCheckpoint(
    filename='last',
    save_on_train_epoch_end=True
     
)

trainer=pl.Trainer(
    default_root_dir=HyperParameter.DEFAULT_ROOT,
    accelerator='gpu',
    callbacks=[checkpoint1,checkpoint2],
    log_every_n_steps=HyperParameter.LOG_INTERVAL,
    deterministic=True,
    max_epochs=HyperParameter.EPOCH,
    num_sanity_val_steps=0

    
)
trainer.fit(model,model.train_loader,model.val_loader)

