import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from sklearn.utils import class_weight

class Loader:
    def __init__(self, data_dir='./dataset'):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, 'bee_data.csv')
        self.image_dir = os.path.join(data_dir, 'bee_imgs', 'bee_imgs')
        self.df = None
        self.label_encoder = LabelEncoder()
        self.class_names = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_csv_data(self):
        self.df = pd.read_csv(self.csv_path)
        
        self.df = self.df[['file', 'health']].copy()
        
        print(f"Dataset size: {self.df.shape}")
        print(f"\nHealth category distribution: {self.df['health'].value_counts()}")
        
        return self.df
    
    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image = image.resize(target_size)
        
        image_array = np.array(image)
        
        return image_array
    
    def load_images_and_labels(self, target_size=(224, 224), sample_size = None):
        if sample_size:
            self.df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        
        images = []
        labels = []
        
        for _, row in self.df.iterrows():
            image_path = os.path.join(self.image_dir, row['file'])
            
            image_array = self.load_and_preprocess_image(image_path, target_size)
            
            images.append(image_array)
            labels.append(row['health'])
        
        print(f"\n{len(images)} images loaded")
        
        X = np.array(images)
        y = np.array(labels)
        
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        return X, y_encoded
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        num_classes = len(self.class_names)
        # For PyTorch CrossEntropyLoss, use integer labels directly
        self.y_train_cat = torch.tensor(self.y_train, dtype=torch.long)
        self.y_val_cat = torch.tensor(self.y_val, dtype=torch.long)
        self.y_test_cat = torch.tensor(self.y_test, dtype=torch.long)
       
        print(f"\nTraining set: {self.X_train.shape[0]}")
        print(f"Validation set: {self.X_val.shape[0]}")
        print(f"Test set: {self.X_test.shape[0]}\n")
        
        train_dist = pd.Series(self.y_train).value_counts().sort_index()
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}: {train_dist.get(i, 0)} samples")
            
        return self.X_train, self.X_val, self.X_test, self.y_train_cat, self.y_val_cat, self.y_test_cat
    
    def calculate_class_weights(self):
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        
        class_weight_dict = dict(enumerate(class_weights))
        
        print("\nClass weights:")
        for i, weight in class_weight_dict.items():
            print(f"{self.class_names[i]}: {weight:.3f}")
            
        return class_weight_dict
    
    def create_data_augmentation(self):
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.2, 0.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transforms, val_transforms
    
    def save_preprocessed_data(self, filename_prefix='bee_health_data'):
        np.save(f'{filename_prefix}_X_train.npy', self.X_train)
        np.save(f'{filename_prefix}_X_val.npy', self.X_val)
        np.save(f'{filename_prefix}_X_test.npy', self.X_test)
        torch.save(self.y_train_cat, f'{filename_prefix}_y_train.pt')
        torch.save(self.y_val_cat, f'{filename_prefix}_y_val.pt')
        torch.save(self.y_test_cat, f'{filename_prefix}_y_test.pt')
        
        import pickle
        with open(f'{filename_prefix}_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)