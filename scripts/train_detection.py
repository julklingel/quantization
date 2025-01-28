import torch.optim as optim

def train_detection(net, trainloader, valloader, device, num_epochs, patience=3):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, (images, targets) in enumerate(trainloader, 0):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = net(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
            if i % 50 == 49:
                actual_loss = running_loss / 50
                print(f'Epoch: {epoch+1}, Mini-Batches: {i+1}, Loss: {actual_loss:.3f}')
                running_loss = 0.0

                if actual_loss < best_loss:
                    best_loss = actual_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    return net

    return net