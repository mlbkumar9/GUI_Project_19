import cv2
import numpy as np
from PIL import Image
import os

def load_and_preprocess_image(image_path, target_size=(300, 300)):
    """Load and preprocess image for display"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize image for display
        image = image.resize(target_size)
        
        return image
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")

def create_overlay_image(original_image, binary_mask, category, damage_area):
    """Create overlay image with damage highlighted"""
    overlay_image = original_image.copy()
    
    # Find contours and draw them
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2)
    
    # Add text annotations
    cv2.putText(overlay_image, f"Category: {category}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay_image, f"Area: {damage_area} px", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return overlay_image

def analyze_damage_opencv(image_path):
    """
    Analyze damage using OpenCV method (white pixel detection)
    This is the original method from your damage_analyzer.py
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, None, None

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, damage_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
        damage_area = cv2.countNonZero(damage_mask)

        return damage_area, damage_mask, image

    except Exception as e:
        print(f"Error in OpenCV analysis: {e}")
        return None, None, None

def save_results(original_image, binary_mask, category, damage_area, filename, output_dir):
    """Save mask and overlay results"""
    try:
        # Create output directories
        mask_dir = os.path.join(output_dir, 'Masks')
        overlay_dir = os.path.join(output_dir, 'Overlays')
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        
        # Save mask
        mask_path = os.path.join(mask_dir, filename)
        cv2.imwrite(mask_path, binary_mask)
        
        # Create and save overlay
        overlay_image = create_overlay_image(original_image, binary_mask, category, damage_area)
        overlay_path = os.path.join(overlay_dir, filename)
        cv2.imwrite(overlay_path, overlay_image)
        
        return mask_path, overlay_path
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return None, None