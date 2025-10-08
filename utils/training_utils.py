import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from keras_unet_collection import models
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import glob
import copy
from threading import Thread
import time

class DamageDataset(Dataset):
    """PyTorch Dataset for damage detection"""
    def __init__(self, image_paths, mask_paths, target_size=(512, 512)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)

        image = image.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1).astype(np.float32) / 255.0

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        mask = torch.from_numpy(mask.transpose((2, 0, 1)))

        return image, mask

class TrainingManager:
    def __init__(self, progress_callback=None, status_callback=None):
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.is_training = False
        
    def update_status(self, message):
        if self.status_callback:
            self.status_callback(message)
        print(message)
    
    def update_progress(self, current, total):
        if self.progress_callback:
            self.progress_callback(current, total)
    
    def load_data_keras(self, image_dir, mask_dir, target_size=(512, 512)):
        """Load data for Keras training"""
        image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
        mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

        X = []
        y = []

        self.update_status(f"Loading {len(image_files)} images and masks...")
        for i, (img_path, mask_path) in enumerate(zip(image_files, mask_files)):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            X.append(img)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, target_size)
            mask = np.expand_dims(mask, axis=-1)
            y.append(mask)
            
            self.update_progress(i + 1, len(image_files))

        X = np.array(X, dtype='float32') / 255.0
        y = np.array(y, dtype='float32') / 255.0

        return X, y
    
    def train_keras_model(self, image_dir, mask_dir, output_dir, backbone='ResNet50', epochs=30, batch_size=4):
        """Train Keras U-Net model"""
        try:
            self.is_training = True
            os.makedirs(output_dir, exist_ok=True)
            
            # Load data
            X, y = self.load_data_keras(image_dir, mask_dir)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.update_status(f"Data loaded. Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Build model
            self.update_status(f"Building U-Net++ with '{backbone}' backbone...")
            model = models.unet_plus_2d((512, 512, 3), 
                                       filter_num=[64, 128, 256, 512],
                                       n_labels=1, 
                                       stack_num_down=2, 
                                       stack_num_up=2,
                                       activation='ReLU', 
                                       output_activation='Sigmoid',
                                       batch_norm=True, 
                                       pool=True, 
                                       unpool=True, 
                                       backbone=backbone, 
                                       weights='imagenet',
                                       name=f'unet-plus_{backbone}')

            model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])
            
            # Training callbacks
            model_save_path = os.path.join(output_dir, f'kuc_unet-plus_{backbone}.keras')
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-6, mode='max')
            ]
            
            self.update_status("Starting training...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            self.update_status(f"Training complete! Model saved to {model_save_path}")
            return True, model_save_path
            
        except Exception as e:
            self.update_status(f"Training failed: {str(e)}")
            return False, None
        finally:
            self.is_training = False
    
    def train_pytorch_model(self, image_dir, mask_dir, output_dir, backbone='resnet50', epochs=25, batch_size=4):
        """Train PyTorch U-Net model"""
        try:
            self.is_training = True
            os.makedirs(output_dir, exist_ok=True)
            
            # Data loading
            image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
            
            train_images, val_images, train_masks, val_masks = train_test_split(
                image_paths, mask_paths, test_size=0.2, random_state=42
            )
            
            train_dataset = DamageDataset(train_images, train_masks)
            val_dataset = DamageDataset(val_images, val_masks)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            self.update_status(f"Found {len(image_paths)} images. Training on {len(train_dataset)}, validating on {len(val_dataset)}.")
            
            # Model setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.update_status(f"Using device: {device}")
            
            model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            ).to(device)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            
            # Training loop
            best_val_loss = float('inf')
            best_model_wts = None
            
            for epoch in range(epochs):
                if not self.is_training:  # Check for cancellation
                    break
                    
                model.train()
                running_train_loss = 0.0
                for images, masks in train_loader:
                    images, masks = images.to(device), masks.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item() * images.size(0)
                
                epoch_train_loss = running_train_loss / len(train_dataset)

                model.eval()
                running_val_loss = 0.0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        running_val_loss += loss.item() * images.size(0)
                
                epoch_val_loss = running_val_loss / len(val_dataset)
                
                self.update_status(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
                self.update_progress(epoch + 1, epochs)

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            # Save model
            if best_model_wts:
                model_save_path = os.path.join(output_dir, f'smp_unet_{backbone}.pth')
                torch.save(best_model_wts, model_save_path)
                self.update_status(f"Training complete! Best model saved to {model_save_path}")
                self.update_status(f"Best Validation Loss: {best_val_loss:.4f}")
                return True, model_save_path
            else:
                self.update_status("Training did not result in a best model to save.")
                return False, None
                
        except Exception as e:
            self.update_status(f"Training failed: {str(e)}")
            return False, None
        finally:
            self.is_training = False
    
    def stop_training(self):
        """Stop the current training process"""
        self.is_training = False