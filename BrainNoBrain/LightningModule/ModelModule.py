from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import monai
from numpy import average
import torch
import torchmetrics


class BrainClassificationModelModule(L.LightningModule) : 

    def __init__(self,loss_function,learning_rate) : 
        super().__init__()
        self.__model = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=2,pretrained = True)
        self.learning_rate = learning_rate
        self.loss_function = loss_function

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Specificity(task="binary"),
            torchmetrics.Accuracy(task="binary", average="macro"),
            torchmetrics.AUROC(task="binary"),
            torchmetrics.Recall(task="binary")])

        self.__train_metrics = metrics.clone(prefix="train_")
        self.__val_metrics = metrics.clone(prefix="val_")
    
    def forward(self,x) -> Any:
        return self.__model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr = self.learning_rate,weight_decay=0.01 )
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.5,end_factor=1,last_epoch=-1)
        return  [optimizer],[scheduler]
    
    def training_step(self, batch,batch_idx) -> STEP_OUTPUT:
        X,y = batch["img"],batch["label"]
        output = self.__model(X)
        loss = self.loss_function(output,y)
        self.log("Training Loss",loss ,on_step = False, on_epoch = True, prog_bar  =True)
        outputs = torch.stack([monai.transforms.Activations(softmax=True)(i) for i in monai.data.decollate_batch(output,detach = False)])
        y_one_hot = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=2) 
        self.__train_metrics.update(torch.as_tensor(outputs), y_one_hot)
        return loss
    
    def validation_step(self, batch,batch_idx) -> STEP_OUTPUT:
        X,y = batch["img"],batch["label"]
        output = self.__model(X)
        loss = self.loss_function(output,y)
        self.log("Validation Loss", loss, on_step = False, on_epoch = True, prog_bar  =True)
        outputs = torch.stack([monai.transforms.Activations(softmax=True)(i) for i in monai.data.decollate_batch(output,detach = False)])
        y_one_hot = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=2) 
        self.__val_metrics.update(torch.as_tensor(outputs), y_one_hot)
        return loss
    
    def on_train_epoch_end(self):
        log_output = self.__train_metrics.compute()
        self.log_dict(log_output)
        self.__train_metrics.reset()
        
    def on_validation_epoch_end(self):
        log_output = self.__val_metrics.compute()
        self.log_dict(log_output)
        self.__val_metrics.reset()