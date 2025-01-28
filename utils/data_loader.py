import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pycocotools.coco import COCO



def load_data(data_dir, batch_size, shuffle=True, resize_x=28, resize_y=28):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, 'train2017')
    val_dir = os.path.join(data_dir, 'val2017')
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return trainloader, valloader, train_dataset.classes




def load_data_detection(data_dir, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class CocoCatDetection(datasets.CocoDetection):
        def __init__(self, root, annFile, transform=None):
            super(CocoCatDetection, self).__init__(root, annFile, transform)
            self.cat_ids = self.coco.getCatIds(catNms=['cat'])
            self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        
        def __getitem__(self, index):
            img, target = super(CocoCatDetection, self).__getitem__(self.img_ids[index])
            target = [obj for obj in target if obj['category_id'] in self.cat_ids]
            return img, target

    train_dir = os.path.join(data_dir, 'train2017')
    val_dir = os.path.join(data_dir, 'val2017')
    train_annFile = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    val_annFile = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    
    train_dataset = CocoCatDetection(root=train_dir, annFile=train_annFile, transform=transform)
    val_dataset = CocoCatDetection(root=val_dir, annFile=val_annFile, transform=transform)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: tuple(zip(*x)))
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: tuple(zip(*x)))
    
    return trainloader, valloader