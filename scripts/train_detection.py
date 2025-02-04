import torch.optim as optim
import torch
from tqdm import tqdm
import mlflow
import mlflow.pytorch

def train_detection(net, trainloader, valloader, device, num_epochs, patience=5):
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        
        for images, targets in tqdm(trainloader, desc=f'Epoch {epoch+1}'):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = net(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()
            
            train_loss += losses.item()

        val_loss = 0.0
        net.train()
        with torch.no_grad():
            for images, targets in valloader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = net(images, targets)
                val_loss += sum(loss.item() for loss in loss_dict.values())
                for loss_name, loss_value in loss_dict.items():
                    mlflow.log_metric(f"val_{loss_name}", loss_value.item(), step=epoch)

        avg_train_loss = train_loss/len(trainloader)
        avg_val_loss = val_loss/len(valloader)
        mlflow.log_metric("epoch_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("epoch_accuracy", avg_val_loss, step=epoch)
             
        for param_group in optimizer.param_groups:
            mlflow.log_metric("learning_rate", param_group['lr'], step=epoch)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.3f}, Val Loss: {avg_val_loss:.3f}')
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(net.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            net.load_state_dict(torch.load('best_model.pth'))
            break

    return net