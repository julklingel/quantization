import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from models.classification.resnet_classification import Resnet18_Classification
from evaluation.classification.eval_classification import evaluate_model

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from models.classification.resnet_classification import Resnet18_Classification
from evaluation.classification.eval_classification import evaluate_model

def train_classification(net, trainloader, valloader, device, num_epochs, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 50 == 49:
                actual_loss = running_loss / 50
                accuracy = evaluate_model(net, valloader, device)
                
                mlflow.log_metric("loss", actual_loss, step=epoch * len(trainloader) + i)
                mlflow.log_metric("accuracy", accuracy, step=epoch * len(trainloader) + i)
                
                print(f'Epoch: {epoch+1}, Mini-Batches: {i+1}, Loss: {actual_loss:.3f}, Accuracy: {accuracy:.3f}%')
                running_loss = 0.0

                if actual_loss < best_loss:
                    best_loss = actual_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    return net

        mlflow.log_metric("epoch_loss", actual_loss, step=epoch)
        mlflow.log_metric("epoch_accuracy", accuracy, step=epoch)

    return net