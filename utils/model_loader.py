import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from keras_unet_collection import models
import segmentation_models_pytorch as smp
import os
import cv2
import numpy as np

class ModelLoader:
    def __init__(self):
        self.pytorch_model = None
        self.keras_model = None
        self.pytorch_backbone = None
        self.keras_backbone = None
        
    def load_pytorch_model(self, model_path, backbone='resnet50'):
        """Load PyTorch U-Net model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"PyTorch model not found: {model_path}")
            
            # Create model architecture
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pytorch_model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            ).to(device)
            
            # Load weights
            self.pytorch_model.load_state_dict(torch.load(model_path, map_location=device))
            self.pytorch_model.eval()
            self.pytorch_backbone = backbone
            
            return True
        except Exception as e:
            print(f"Error loading PyTorch model: {str(e)}")
            return False
    
    def load_keras_model(self, model_path, backbone='ResNet50'):
        """Load Keras U-Net model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Keras model not found: {model_path}")
            
            # Create model architecture
            self.keras_model = models.unet_plus_2d((512, 512, 3), 
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
                                                   weights=None,
                                                   name=f'unet-plus_{backbone}')
            
            # Load weights
            self.keras_model.load_weights(model_path)
            self.keras_backbone = backbone
            
            return True
        except Exception as e:
            print(f"Error loading Keras model: {str(e)}")
            return False
    
    def predict_pytorch_segmentation(self, image_path, target_size=(512, 512)):
        """Make segmentation prediction using PyTorch model"""
        if self.pytorch_model is None:
            raise ValueError("PyTorch model not loaded")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        original_size = (image.shape[1], image.shape[0])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, target_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        device = next(self.pytorch_model.parameters()).device
        image_tensor = torch.from_numpy(image_normalized.transpose((2, 0, 1))).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.pytorch_model(image_tensor)
            pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0] > 0.5

        binary_mask = (pred_mask).astype(np.uint8) * 255
        binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return binary_mask_resized, image
    
    def predict_keras_segmentation(self, image_path, target_size=(512, 512)):
        """Make segmentation prediction using Keras model"""
        if self.keras_model is None:
            raise ValueError("Keras model not loaded")
        
        # Load and preprocess image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(original_image_rgb, target_size)
        input_image = input_image / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        # Predict
        predicted_mask = self.keras_model.predict(input_image, verbose=0)[0]
        binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
        binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))
        
        return binary_mask_resized, original_image
    
    def analyze_damage(self, binary_mask):
        """Analyze damage from binary mask"""
        damage_area = cv2.countNonZero(binary_mask)
        
        # Thresholds from your original code
        MANAGEABLE_AREA_THRESHOLD = 5026
        PARTIALLY_DAMAGED_AREA_THRESHOLD = 17671
        
        category = "No Damage Detected"
        if damage_area > PARTIALLY_DAMAGED_AREA_THRESHOLD:
            category = "Completely damaged"
        elif damage_area > MANAGEABLE_AREA_THRESHOLD:
            category = "Partially damaged"
        elif damage_area > 0:
            category = "Manageable"
            
        return category, damage_area