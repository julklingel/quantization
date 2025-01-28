import torch
from torchvision import datasets, transforms
import os



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



def load_data_detection(data_dir, batch_size, shuffle=True, resize_x=28, resize_y=28):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, 'train2017', 'cats')
    val_dir = os.path.join(data_dir, 'val2017', 'cats')
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return 
    
    
    
    