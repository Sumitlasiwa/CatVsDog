import torch
import torch.nn as nn 
from utils.dataset import get_dataloaders
from utils.train_eval import train_model
from models.alexnet import AlexNet
from models.vgg16 import Vgg16
from models.resent import ResNet18
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train CNN for Cat vs Dog Classification")
    
    parser.add_argument('-m','--model', type=str, choices=['alexnet', 'vgg16', 'resnet18'], required=True, help='Model type to train')
    parser.add_argument('-e','--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('-bs','--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-lr','--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-p','--patience', type= int, default=None, help='Use early stopping if patience is set')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "alexnet":
        model = AlexNet(num_classes=1).to(device)
        img_size = 227
    elif args.model == "vgg16":
        model = Vgg16(num_classes=1).to(device)
        img_size = 224
    elif args.model == "resnet18":
        model = ResNet18(num_classes=1).to(device)
        img_size = 224
    else:
        raise ValueError("Invalid model name")
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            data_dir="data/",
            batch_size=args.batch_size, 
            img_size=img_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training {args.model} for {args.epochs} epochs | Batch size: {args.batch_size} | LR: {args.lr} | Early stopping with patience: {args.patience}")
    print(f"Runnin on {(torch.cuda.get_device_name()) if torch.cuda.is_available() else print("CPU")}")

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device=device, num_epochs=args.epochs, model_name=args.model, patience = args.patience)

if __name__ == '__main__':
    main()
