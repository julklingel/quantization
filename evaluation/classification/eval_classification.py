import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model(net, testloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def plot_confusion_matrix(net, testloader, class_names, device, save_path="confusion_matrix.png"):
    y_pred, y_true, _, _, _ = get_all_preds(net, testloader, device)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    mlflow.log_artifact(save_path)
    plt.close()

def get_all_preds(model, loader, device):
    all_preds = []
    all_labels = []
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1]
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            misclassified_mask = predicted.cpu() != labels
            misclassified_images.extend(images[misclassified_mask].cpu())
            misclassified_labels.extend(labels[misclassified_mask].cpu().numpy())
            misclassified_preds.extend(predicted[misclassified_mask].cpu().numpy())
    return np.array(all_preds), np.array(all_labels), misclassified_images, misclassified_labels, misclassified_preds

def plot_misclassified_images(images, true_labels, predicted_labels, class_names, num_images=5):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(f'True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}')
        plt.axis('off')
    plt.show()