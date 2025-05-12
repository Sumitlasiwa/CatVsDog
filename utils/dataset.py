import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import  ImageFolder

def get_transforms(img_size=224):
    transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean= [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    return transform
    
def get_dataloaders(data_dir, batch_size=32, img_size=224):
    transform = get_transforms(img_size)
    
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle= True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle= True, num_workers=4, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader
    
    