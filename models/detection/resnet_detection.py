

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops import generalized_box_iou_loss, box_iou
import torch
import torch.nn as nn

class Resnet18_Detection(nn.Module):
    def __init__(self, optimizer_type='Adam'):
        super().__init__()
        
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights, progress=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4) 
        
        self.optimizer_type = optimizer_type

        self.mse_loss_fn = nn.MSELoss()
        self.iou_loss_fn = generalized_box_iou_loss  

        self.mse_loss_weight = 0.2
        self.iou_loss_weight = 0.8

        if optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        elif optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        elif optimizer_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-4)
        else: 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def calculate_iou(self, preds, targets):
        preds_boxes = [torch.tensor([p[0], p[1], p[2], p[3]]) for p in preds]
        targets_boxes = [torch.tensor([t[0], t[1], t[2], t[3]]) for t in targets]
        ious = box_iou(torch.stack(preds_boxes), torch.stack(targets_boxes))
        return ious

    def calculate_f1_score(self, ious, iou_threshold=0.5):
        true_positives = ious.diag().ge(iou_threshold).sum().item()
        false_positives = len(ious) - true_positives
        false_negatives = len(ious) - true_positives

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1
    
    def compute_loss(self, pred, label):
        mse_loss = self.mse_loss_fn(pred, label)
        iou_loss = self.iou_loss_fn(pred, label, reduction='mean')

        loss = self.mse_loss_weight * mse_loss + self.iou_loss_weight * iou_loss
        return loss, mse_loss, iou_loss
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        x_ray = x_ray.float()
        label = label.float()
        pred = self(x_ray)
        loss, mse_loss, iou_loss = self.compute_loss(pred, label)
        
        self.log("Train Loss/Combined", loss)
        self.log("Train Loss/MSE", mse_loss)
        self.log("Train Loss/IoU", iou_loss)
        
        if batch_idx % 50 == 0:
            self.log_images(x_ray.cpu(), pred.cpu(), label.cpu(), "Train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        x_ray = x_ray.float()
        pred = self(x_ray)
        loss, mse_loss, iou_loss = self.compute_loss(pred, label)
        
        self.log("Val Loss", loss)
        self.log("Val Loss/MSE", mse_loss)
        self.log("Val Loss/IoU", iou_loss)
        
        ious = self.calculate_iou(pred, label)
        f1_score = self.calculate_f1_score(ious)
        
        self.log('val_f1', f1_score, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_f1': f1_score}
    

        
    def configure_optimizers(self):
        return [self.optimizer]