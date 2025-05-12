import torch
import os
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter
import datetime

def train_one_step(model, criteria, optimizer, images, labels):
    model.train()
    optimizer.zero_grad()
    
    #Forward Pass
    results = model(images)
    results = results.squeeze(1)
    labels = labels.type(torch.float)
    loss = criteria(results, labels)
    
    #backward Pass
    loss.backward()
    
    # Gradient Clipping 
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    #Update weights after clipping
    optimizer.step()
    
    return loss, results

def val_one_step(model, criteria, images, labels):
    model.eval()
    
    with torch.no_grad():
        #Forward Pass
        results = model(images)
        results = results.squeeze(1)
        labels = labels.type(torch.float)
        loss = criteria(results, labels)
        
    return loss, results

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, model_name, patience):
    log_dir = os.path.join("runs", model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    log_data = {     #log metrics
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    save_path = os.path.join("results", model_name)
    os.makedirs(save_path, exist_ok=True)
    #Training
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        train_epoch_loss = 0
        train_epoch_correct_prediction = 0
        for train_images, train_labels in tqdm(train_dataloader, desc=f'Training {epoch+1} of {num_epochs}'):
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            loss, results = train_one_step(model, criterion, optimizer, train_images, train_labels)
            train_epoch_loss += loss.item()
            results = torch.sigmoid(results).round()
    
            correct_prediction = (results == train_labels).type(torch.float).sum()
            train_epoch_correct_prediction += correct_prediction.item()

        train_epoch_loss /= len(train_dataloader)
        log_data["train_loss"].append(train_epoch_loss)
        writer.add_scalar("Loss/train",train_epoch_loss, epoch)

        epoch_train_acc = train_epoch_correct_prediction / len(train_dataloader.dataset)
        log_data["train_acc"].append(epoch_train_acc)
        writer.add_scalar("Accuracy/train",epoch_train_acc, epoch)

        
        #Validation
        val_epoch_loss = 0
        val_epoch_correct_prediction = 0
        for val_images, val_labels in tqdm(val_dataloader, desc=f'Validation'):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            loss, results = val_one_step(model, criterion, val_images, val_labels)
            val_epoch_loss += loss.item()
            results = torch.sigmoid(results).round()
            
            correct_prediction = (results == val_labels).type(torch.float).sum()
            val_epoch_correct_prediction += correct_prediction.item()

        val_epoch_loss /= len(val_dataloader)
        log_data["val_loss"].append(val_epoch_loss)
        writer.add_scalar("Loss/val",val_epoch_loss, epoch)


        epoch_val_acc = val_epoch_correct_prediction / len(val_dataloader.dataset)
        log_data["val_acc"].append(epoch_val_acc)
        writer.add_scalar("Accuracy/val",epoch_val_acc, epoch)
        
        print(f"Epoch {epoch+1} of {num_epochs}:")
        print(f"Train Acc: {epoch_train_acc:.4f}, Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Acc: {epoch_val_acc:.4f}, Val Loss: {val_epoch_loss:.4f}")

        # Early stopping & best model saving
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print("✅ Saved new best model!")
            counter = 0
        else:
            if patience is not None:
                counter += 1
                if counter >= patience:
                    print("⏹️ Early stopping triggered.")
                    break
            
        # save log data to folder results and to respective model folder for future use  
        with open(os.path.join(save_path, "log_data.json"), "w") as f:
            json.dump(log_data, f) 
            
    writer.close()
              
    print(f"Best validation loss: {best_val_loss:.4f}")
    

    
    