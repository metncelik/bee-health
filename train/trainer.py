import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class BeeHealthDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
        return image, label


class BeeHealthCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(BeeHealthCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        
        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def create_model(model_type='cnn', num_classes=6, pretrained=True):
    if model_type.lower() == 'cnn':
        return BeeHealthCNN(num_classes=num_classes)
    
    # elif model_type.lower() == 'resnet50':
    #     model = models.resnet50(pretrained=pretrained)
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)
    #     return model
    
    elif model_type.lower() == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model


class Trainer:
    def __init__(self, loader, model_type='cnn', num_classes=6, learning_rate=0.001, batch_size=32, pretrained=True):
        self.loader = loader
        self.model_type = model_type
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = create_model(model_type=model_type, num_classes=num_classes, pretrained=pretrained).to(self.device)
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        self.save_dir = f'best_models/{model_type}_{time.strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.save_dir, exist_ok=True)
        
    def setup_data_loaders(self, class_weights=None):
        train_transforms, val_transforms = self.loader.create_data_augmentation()
        
        train_dataset = BeeHealthDataset(self.loader.X_train, self.loader.y_train_cat, transform=train_transforms)
        val_dataset = BeeHealthDataset(self.loader.X_val, self.loader.y_val_cat, transform=val_transforms)
        test_dataset = BeeHealthDataset(self.loader.X_test, self.loader.y_test_cat, transform=val_transforms)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        if class_weights:
            weights = torch.tensor([class_weights[i] for i in range(self.num_classes)], dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for _, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=50, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on device: {self.device}")
        print(f"Model type: {self.model_type.upper()}")
        print(f"Using pretrained weights: {self.pretrained}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        print("\nTraining completed!")
        
    def evaluate(self, model_path=None):
        if model_path:
            self.load_model(model_path)
            
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        print("\nTest Set Evaluation:")
        print("-" * 40)
        print(classification_report(y_true, y_pred, target_names=self.loader.class_names))
        
        return y_true, y_pred
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['train_loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(self.history['train_acc'], label='Training Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/{self.save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.loader.class_names,
                    yticklabels=self.loader.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'plots/{self.save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'class_names': self.loader.class_names,
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained
        }, f"{self.save_dir}/best_model.pth")
        
    def load_model(self, filename):
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
            num_classes = checkpoint.get('num_classes', self.num_classes)
            pretrained = checkpoint.get('pretrained', False)
            self.model = create_model(model_type=model_type, num_classes=num_classes, pretrained=pretrained).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']