# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
from torchvision import transforms, models
from PIL import Image
import os
import multiprocessing

# %%
# Custom dataset for Food101
class Food101Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes = sorted(os.listdir(os.path.join(root_dir, split)))
        self.image_paths = []
        self.labels = []
        
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, split, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(i)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# %%
# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Data augmentation
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
])

# %%
# Load a pre-trained MobileNetV2 model
base_model = models.mobilenet_v2(pretrained=True)

# %%
# Fine-tuning: Unfreeze the top layers of the base model
for param in base_model.parameters():
    param.requires_grad = False

# Unfreeze the last 100 layers
for param in base_model.features[-100:].parameters():
    param.requires_grad = True

# %%
# Create the model
class FoodRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FoodRecognitionModel, self).__init__()
        self.base_model = base_model
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_model.last_channel, num_classes)
        )
        self.data_augmentation = data_augmentation
    
    def forward(self, x):
        if self.training:
            x = self.data_augmentation(x)
        return self.base_model(x)

# %%
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    print("Num GPUs Available: ", torch.cuda.device_count())
    print(torch.__version__)
    print(torch.cuda.is_available())

    # Load Fruits-360 dataset
    fruits360_train = torchvision.datasets.ImageFolder(
        'fruits-360_dataset_100x100/fruits-360/Training',
        transform=transform
    )

    fruits360_test = torchvision.datasets.ImageFolder(
        'fruits-360_dataset_100x100/fruits-360/Test',
        transform=transform
    )

    # Combine the datasets
    train_dataset = fruits360_train
    test_dataset = fruits360_test

    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_classes = len(fruits360_train.classes)
    model = FoodRecognitionModel(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%')

    # Save the model
    torch.save(model.state_dict(), 'food_model_combined_v1.pth')
