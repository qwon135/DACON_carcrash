import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import rearrange
from decord import VideoReader
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch.losses import FocalLoss
from transformers import AutoModel, AutoImageProcessor, AutoConfig
from skmultilearn.model_selection import iterative_train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorchvideo.transforms.transforms_factory import create_video_transform

from crash_modules.crash_dataset import VideoDataset

class Timing(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = config['learning_rate']
        self.model = AutoModel.from_pretrained('facebook/timesformer-base-finetuned-k600')
        self.classifier = nn.LazyLinear(config['n_classes'])
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = FocalLoss('multiclass')

    def forward(self, x):
        x = self.model(x).last_hidden_state.mean(dim=1)
        x_out = self.classifier(x)
        return x_out.T[0]

    def training_step(self, batch, batch_idx, optimizer_idx):
        video, label, label_split = batch['video'], batch['label'], batch['label_split']
        y_hats = self.forward(batch["video"]).float()
        loss = self.loss(y_hats, batch["label"].float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        video, label, label_split = batch['video'], batch['label'], batch['label_split']
        y_hats = self.forward(batch["video"]).float()

        with torch.no_grad():
            loss = self.loss(y_hats, batch["label"].float())

        self.log("valid_loss", loss)

        step_output = [y_hats, label]
        return step_output
    
    
    def predict_step(self, batch, batch_idx):
        video, _, _ = batch['video'], batch['label'], batch['label_split']
        y_hats = self.forward(batch["video"])
        step_output = y_hats
        return step_output

    def validation_epoch_end(self, step_outputs):
        preds = []
        labels = []

        for step_output in step_outputs:
            pred, label = step_output
            preds += pred.detach().cpu().tolist()
            labels += label.tolist()            
        preds = np.array(preds)        
        preds = np.where(preds>0.5, 1, 0)    
        score = f1_score(labels, preds)
        self.log("val_score", score)
        
        return score
    
    
    def post_preproc(self, step_outputs):
        preds = []
        for step_output in step_outputs:
            preds += step_output.argmax(1).detach().cpu().tolist()
        return preds
    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.parameters(), lr=self.learning_rate)        
        opt2 = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        optimizers = [opt1, opt2]

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, mode='min', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)        
        lr_schedulers = {"scheduler": scheduler, "monitor": "valid_loss"}
        
        return optimizers, lr_schedulers