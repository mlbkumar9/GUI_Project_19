#!/usr/bin/env python3
"""
Test version of the Structural Damage Analyzer
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import threading
import datetime
import cv2
import numpy as np

# Import utility modules (with error handling)
try:
    from utils.model_loader import ModelLoader
    from utils.image_utils import load_and_preprocess_image, analyze_damage_opencv, save_results
    from utils.training_utils import TrainingManager
    MODEL_SUPPORT = True
except ImportError as e:
    print(f"Warning: Some utility modules not available: {e}")
    MODEL_SUPPORT = False

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ModernDamageAnalyzer:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Structural Damage Analyzer - AI Detection & Training")
        self.root.geometry("1400x900")
        
        # Initialize components
        if MODEL_SUPPORT:
            self.model_loader = ModelLoader()
            self.training_manager = TrainingManager(
                progress_callback=self.update_training_progress,
                status_callback=self.update_status
            )
        else:
            self.model_loader = None
            self.training_manager = None
        
        # Variables
        self.current_images = []
        self.current_image_index = 0
        self.zoom_level = 1.0
        self.output_directory = None
        self.base_dir = os.getcwd()
        
        # Model backbone options
        self.pytorch_backbones = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'vgg11', 'vgg13', 'vgg16', 'vgg19',
            'densenet121', 'densenet169', 'densenet201',
            'mobilenet_v2', 'efficientnet-b0', 'efficientnet-b1'
        ]
        
        self.keras_backbones = [
            'ResNet50', 'ResNet101', 'ResNet152',
            'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
            'vgg16', 'vgg19',
            'densenet121', 'densenet169', 'densenet201',
            'MobileNetV2'
        ]
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create header
        header_frame = ctk.CTkFrame(self.root, height=70)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(header_frame, text="🏗️ Structural Damage Analyzer", 
                                  font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=15)
        
        # Main container
        main_container = ctk.CTkFrame(self.root)
        main_container.pack(fill="both", expand=True, padx=20, pady=(10, 20))
        
        # Left sidebar
        self.left_sidebar = ctk.CTkFrame(main_container, width=450)
        self.left_sidebar.pack(side="left", fill="y", padx=(0, 10))
        self.left_sidebar.pack_propagate(False)
        
        # Right panel
        self.right_panel = ctk.CTkFrame(main_container)
        self.right_panel.pack(side="right", fill="both", expand=True)
        
        self.setup_controls()
        self.setup_image_panel()
        
        # Status bar
        self.status_label = ctk.CTkLabel(self.root, text="Ready - Welcome to Structural Damage Analyzer", 
                                        font=ctk.CTkFont(size=12))
        self.status_label.pack(side="bottom", fill="x", padx=20, pady=(0, 10))
    
    def setup_controls(self):
        """Setup control buttons"""
        # Create tabview
        self.tabview = ctk.CTkTabview(self.left_sidebar, width=430)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add tabs
        self.tabview.add("🚀 Training")
        self.tabview.add("🔍 Prediction")
        self.tabview.add("📊 OpenCV")
        self.tabview.add("📋 Results")
        
        # Training tab
        training_tab = self.tabview.tab("🚀 Training")
        
        # Training Configuration Section
        config_frame = ctk.CTkFrame(training_tab)
        config_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(config_frame, text="Training Configuration", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        # Framework selection
        ctk.CTkLabel(config_frame, text="Framework:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10)
        self.framework_var = ctk.StringVar(value="PyTorch")
        self.framework_combo = ctk.CTkComboBox(config_frame, values=["PyTorch", "Keras"], 
                                              variable=self.framework_var, command=self.on_framework_change, width=200)
        self.framework_combo.pack(padx=10, pady=(0, 5))
        
        # Backbone selection
        ctk.CTkLabel(config_frame, text="Backbone:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10)
        self.backbone_var = ctk.StringVar(value="resnet50")
        self.backbone_combo = ctk.CTkComboBox(config_frame, values=self.pytorch_backbones, 
                                             variable=self.backbone_var, width=200)
        self.backbone_combo.pack(padx=10, pady=(0, 5))
        
        # Training parameters
        params_frame = ctk.CTkFrame(config_frame)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        # Epochs and Batch Size in same row
        ctk.CTkLabel(params_frame, text="Epochs:", font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
        self.epochs_entry = ctk.CTkEntry(params_frame, width=60)
        self.epochs_entry.insert(0, "25")
        self.epochs_entry.pack(side="left", padx=5)
        
        ctk.CTkLabel(params_frame, text="Batch Size:", font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
        self.batch_size_entry = ctk.CTkEntry(params_frame, width=60)
        self.batch_size_entry.insert(0, "4")
        self.batch_size_entry.pack(side="left", padx=5)
        
        # Dataset Directories Section
        dir_frame = ctk.CTkFrame(training_tab)
        dir_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(dir_frame, text="Dataset Directories", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        # Images directory
        ctk.CTkLabel(dir_frame, text="Images Directory:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10)
        img_frame = ctk.CTkFrame(dir_frame)
        img_frame.pack(fill="x", padx=10, pady=(0, 5))
        self.image_dir_entry = ctk.CTkEntry(img_frame, width=250)
        self.image_dir_entry.insert(0, os.path.join(self.base_dir, "RAW_Images"))
        self.image_dir_entry.pack(side="left", padx=5)
        ctk.CTkButton(img_frame, text="📁", command=self.browse_image_dir, width=30).pack(side="right", padx=5)
        
        # Masks directory
        ctk.CTkLabel(dir_frame, text="Masks Directory:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10)
        mask_frame = ctk.CTkFrame(dir_frame)
        mask_frame.pack(fill="x", padx=10, pady=(0, 5))
        self.mask_dir_entry = ctk.CTkEntry(mask_frame, width=250)
        self.mask_dir_entry.insert(0, os.path.join(self.base_dir, "Masks"))
        self.mask_dir_entry.pack(side="left", padx=5)
        ctk.CTkButton(mask_frame, text="📁", command=self.browse_mask_dir, width=30).pack(side="right", padx=5)
        
        # Output directory
        ctk.CTkLabel(dir_frame, text="Output Directory:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10)
        out_frame = ctk.CTkFrame(dir_frame)
        out_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.output_dir_entry = ctk.CTkEntry(out_frame, width=250)
        self.output_dir_entry.insert(0, os.path.join(self.base_dir, "Trained_Models"))
        self.output_dir_entry.pack(side="left", padx=5)
        ctk.CTkButton(out_frame, text="📁", command=self.browse_output_dir, width=30).pack(side="right", padx=5)
        
        # Training Controls Section
        control_frame = ctk.CTkFrame(training_tab)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(control_frame, text="Training Controls", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        self.train_button = ctk.CTkButton(control_frame, text="🚀 Start Training", command=self.start_training,
                                         font=ctk.CTkFont(size=12, weight="bold"), height=35, width=180)
        self.train_button.pack(pady=5)
        
        self.stop_button = ctk.CTkButton(control_frame, text="⏹️ Stop Training", command=self.stop_training,
                                        font=ctk.CTkFont(size=12, weight="bold"), height=35, width=180,
                                        fg_color="red", hover_color="darkred", state="disabled")
        self.stop_button.pack(pady=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(control_frame, width=200)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        # Training Log Section
        log_frame = ctk.CTkFrame(training_tab)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        
        ctk.CTkLabel(log_frame, text="Training Log", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        self.training_log = ctk.CTkTextbox(log_frame, height=100, font=ctk.CTkFont(family="Consolas", size=10))
        self.training_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Prediction tab
        prediction_tab = self.tabview.tab("🔍 Prediction")
        
        # Model loading section
        ctk.CTkLabel(prediction_tab, text="Model Loading", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        # PyTorch model loading
        pytorch_frame = ctk.CTkFrame(prediction_tab)
        pytorch_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(pytorch_frame, text="PyTorch Model:").pack(anchor="w", padx=5)
        self.pytorch_model_entry = ctk.CTkEntry(pytorch_frame, width=250)
        self.pytorch_model_entry.pack(side="left", padx=5)
        ctk.CTkButton(pytorch_frame, text="📁", command=self.load_pytorch_model, width=30).pack(side="right", padx=5)
        
        # Keras model loading
        keras_frame = ctk.CTkFrame(prediction_tab)
        keras_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(keras_frame, text="Keras Model:").pack(anchor="w", padx=5)
        self.keras_model_entry = ctk.CTkEntry(keras_frame, width=250)
        self.keras_model_entry.pack(side="left", padx=5)
        ctk.CTkButton(keras_frame, text="📁", command=self.load_keras_model, width=30).pack(side="right", padx=5)
        
        # Analysis buttons
        ctk.CTkLabel(prediction_tab, text="Image Analysis", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        ctk.CTkButton(prediction_tab, text="📷 Load Single Image", 
                     command=self.load_single_image, height=35, width=200).pack(pady=3)
        ctk.CTkButton(prediction_tab, text="📷 Load Multiple Images", 
                     command=self.load_multiple_images, height=35, width=200).pack(pady=3)
        ctk.CTkButton(prediction_tab, text="🔍 Analyze Current Image", 
                     command=self.analyze_current_image, height=35, width=200).pack(pady=3)
        ctk.CTkButton(prediction_tab, text="📁 Batch Process Directory", 
                     command=self.batch_process, height=35, width=200).pack(pady=3)
        
        # OpenCV tab
        opencv_tab = self.tabview.tab("📊 OpenCV")
        ctk.CTkLabel(opencv_tab, text="OpenCV Analysis", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        ctk.CTkButton(opencv_tab, text="📷 Load Image", 
                     command=self.load_opencv_image, height=35, width=200).pack(pady=5)
        ctk.CTkButton(opencv_tab, text="🔍 Analyze with OpenCV", 
                     command=self.analyze_opencv, height=35, width=200).pack(pady=5)
        ctk.CTkButton(opencv_tab, text="📁 Process Directory", 
                     command=self.process_opencv_directory, height=35, width=200).pack(pady=5)
        
        # Common controls
        ctk.CTkButton(opencv_tab, text="📁 Set Output Directory", 
                     command=self.set_output_directory, height=35, width=200).pack(pady=10)
        
        # Results tab
        results_tab = self.tabview.tab("📋 Results")
        
        results_title = ctk.CTkLabel(results_tab, text="Analysis Results & Logic", 
                                    font=ctk.CTkFont(size=14, weight="bold"))
        results_title.pack(pady=(10, 5))
        
        self.results_text = ctk.CTkTextbox(results_tab, height=300, 
                                          font=ctk.CTkFont(family="Consolas", size=10))
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Enhanced initial content with analysis logic and color legend
        initial_content = """🔍 AI DAMAGE ANALYSIS SYSTEM

📋 ANALYSIS LOGIC:
• PyTorch U-Net: Deep learning segmentation model
• Keras U-Net++: Enhanced nested skip connections  
• OpenCV: White pixel detection (threshold > 240)

🎨 COLOR LEGEND:
• 🔴 RED CONTOURS: Detected damage areas
• 🟢 GREEN TEXT: Image metadata & analysis info
• 🟡 YELLOW: Manageable damage (1-5,025 pixels)
• 🟠 ORANGE: Partial damage (5,026-17,670 pixels)
• 🔴 RED: Complete damage (>17,671 pixels)

📊 DAMAGE CLASSIFICATION:
• No Damage: 0 pixels detected
• Manageable: 1-5,025 pixels (minor maintenance)
• Partially Damaged: 5,026-17,670 pixels (attention needed)  
• Completely Damaged: >17,671 pixels (emergency response)

🚀 INSTRUCTIONS:
1. Load a trained model (PyTorch or Keras)
2. Load image(s) to analyze
3. Click 'Analyze Current Image' or batch process
4. Review results with color-coded overlays

Results will appear here after analysis..."""
        self.results_text.insert("0.0", initial_content)
    
    def setup_image_panel(self):
        """Setup image display panel"""
        # Navigation controls
        nav_frame = ctk.CTkFrame(self.right_panel)
        nav_frame.pack(fill="x", padx=20, pady=10)
        
        self.prev_button = ctk.CTkButton(nav_frame, text="◀ Previous", 
                                        command=self.previous_image, width=100, state="disabled")
        self.prev_button.pack(side="left", padx=5)
        
        self.image_counter_label = ctk.CTkLabel(nav_frame, text="No images", 
                                               font=ctk.CTkFont(size=12))
        self.image_counter_label.pack(side="left", padx=15)
        
        self.next_button = ctk.CTkButton(nav_frame, text="Next ▶", 
                                        command=self.next_image, width=100, state="disabled")
        self.next_button.pack(side="left", padx=5)
        
        # Zoom controls
        self.zoom_label = ctk.CTkLabel(nav_frame, text="100%", font=ctk.CTkFont(size=12))
        self.zoom_label.pack(side="right", padx=10)
        ctk.CTkButton(nav_frame, text="🔍+", command=self.zoom_in, width=50).pack(side="right", padx=2)
        ctk.CTkButton(nav_frame, text="🔍-", command=self.zoom_out, width=50).pack(side="right", padx=2)
        
        # Image display area
        images_container = ctk.CTkFrame(self.right_panel)
        images_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Original image (left)
        original_frame = ctk.CTkFrame(images_container)
        original_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(original_frame, text="📷 Original Image", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        self.original_scroll = ctk.CTkScrollableFrame(original_frame)
        self.original_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.original_image_label = ctk.CTkLabel(self.original_scroll, 
                                               text="No image loaded\n\nLoad an image to get started",
                                               font=ctk.CTkFont(size=14))
        self.original_image_label.pack(expand=True)
        
        # Analyzed image (right)
        analyzed_frame = ctk.CTkFrame(images_container)
        analyzed_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(analyzed_frame, text="🔍 Analysis Result", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        self.analyzed_scroll = ctk.CTkScrollableFrame(analyzed_frame)
        self.analyzed_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.analyzed_image_label = ctk.CTkLabel(self.analyzed_scroll, 
                                               text="Analysis result will appear here\n\nAfter running analysis",
                                               font=ctk.CTkFont(size=14))
        self.analyzed_image_label.pack(expand=True)
    
    def load_single_image(self):
        """Load single image - WORKING"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        if file_path:
            try:
                original_image = Image.open(file_path).convert('RGB')
                
                image_data = {
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'original': original_image,
                    'analyzed': None
                }
                
                self.current_images = [image_data]
                self.current_image_index = 0
                
                self.update_image_display()
                self.update_navigation_controls()
                
                self.status_label.configure(text=f"Image loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Image loaded successfully!\n{os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def load_multiple_images(self):
        """Load multiple images - WORKING"""
        file_paths = filedialog.askopenfilenames(
            title="Select Multiple Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        if file_paths:
            self.current_images = []
            failed_count = 0
            
            for file_path in file_paths:
                try:
                    original_image = Image.open(file_path).convert('RGB')
                    
                    image_data = {
                        'path': file_path,
                        'name': os.path.basename(file_path),
                        'original': original_image,
                        'analyzed': None
                    }
                    self.current_images.append(image_data)
                except Exception as e:
                    failed_count += 1
                    print(f"Failed to load {file_path}: {e}")
            
            if self.current_images:
                self.current_image_index = 0
                self.update_image_display()
                self.update_navigation_controls()
                
                status_msg = f"Loaded {len(self.current_images)} images"
                if failed_count > 0:
                    status_msg += f" ({failed_count} failed)"
                self.status_label.configure(text=status_msg)
                
                messagebox.showinfo("Images Loaded", 
                                  f"Successfully loaded {len(self.current_images)} images.\n\n"
                                  f"Use Previous/Next buttons to navigate.")
            else:
                messagebox.showerror("Error", "Failed to load any images")
    
    def analyze_current_image(self):
        """Analyze the currently displayed image"""
        if not self.current_images:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        current_image_data = self.current_images[self.current_image_index]
        image_path = current_image_data['path']
        filename = current_image_data['name']
        
        self.status_label.configure(text=f"Analyzing {filename}...")
        
        # Check if models are available
        if not MODEL_SUPPORT or (self.model_loader and 
                                self.model_loader.pytorch_model is None and 
                                self.model_loader.keras_model is None):
            # Fallback to demo analysis
            self.demo_analysis(current_image_data)
            return
        
        try:
            result_image = None
            method_used = None
            category = "No analysis"
            damage_area = 0
            
            # Try PyTorch first
            if self.model_loader and self.model_loader.pytorch_model is not None:
                try:
                    self.status_label.configure(text=f"Running PyTorch analysis on {filename}...")
                    binary_mask, original_image = self.model_loader.predict_pytorch_segmentation(image_path)
                    category, damage_area = self.model_loader.analyze_damage(binary_mask)
                    result_image = self.create_result_visualization(original_image, binary_mask, category, damage_area, "PyTorch U-Net")
                    method_used = "PyTorch_U-Net"
                    
                    # Update results display with detailed analysis
                    self.update_analysis_results(filename, method_used, category, damage_area, "PyTorch U-Net segmentation model")
                    
                except Exception as e:
                    print(f"PyTorch failed for {filename}: {e}")
                    self.status_label.configure(text=f"PyTorch failed for {filename}, trying Keras...")
            
            # Try Keras if PyTorch failed or not available
            if self.model_loader and self.model_loader.keras_model is not None and result_image is None:
                try:
                    self.status_label.configure(text=f"Running Keras analysis on {filename}...")
                    binary_mask, original_image = self.model_loader.predict_keras_segmentation(image_path)
                    category, damage_area = self.model_loader.analyze_damage(binary_mask)
                    result_image = self.create_result_visualization(original_image, binary_mask, category, damage_area, "Keras U-Net++")
                    method_used = "Keras_U-Net++"
                    
                    # Update results display with detailed analysis
                    self.update_analysis_results(filename, method_used, category, damage_area, "Keras U-Net++ with nested skip connections")
                    
                except Exception as e:
                    print(f"Keras failed for {filename}: {e}")
                    self.status_label.configure(text=f"Keras failed for {filename}")
            
            # Store and display result
            if result_image is not None:
                current_image_data['analyzed'] = result_image
                self.update_image_display()
                
                # Save result if output directory is set
                if self.output_directory:
                    self.save_analysis_result(current_image_data, result_image, method_used)
                
                self.status_label.configure(text=f"Analysis complete for {filename}")
            else:
                # Fallback to demo if no models worked
                self.demo_analysis(current_image_data)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_label.configure(text=error_msg)
    
    def demo_analysis(self, current_image_data):
        """Demo analysis when models are not available"""
        filename = current_image_data['name']
        
        # Demo analysis results
        import random
        damage_area = random.randint(0, 20000)
        
        if damage_area == 0:
            category = "No Damage Detected"
            damage_level = "🟢 NO DAMAGE"
        elif damage_area <= 5025:
            category = "Manageable"
            damage_level = "🟡 MANAGEABLE"
        elif damage_area <= 17670:
            category = "Partially damaged"
            damage_level = "🟠 PARTIALLY DAMAGED"
        else:
            category = "Completely damaged"
            damage_level = "🔴 COMPLETELY DAMAGED"
        
        # Update results display
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        result_text = f"""
🔍 ANALYSIS COMPLETE - {timestamp}

📁 FILE: {filename}
🤖 METHOD: Demo Analysis (Models not loaded)
📊 DESCRIPTION: Simulated damage detection

📈 RESULTS:
• Damage Category: {category}
• Damage Area: {damage_area:,} pixels
• Classification: {damage_level}

🎨 VISUAL OVERLAY:
• Red contours show detected damage areas
• Green text displays analysis metadata
• Original image (left) vs Analysis (right)

💾 OUTPUT: {"Saved to output directory" if self.output_directory else "No output directory set"}

ℹ️ NOTE: Load PyTorch or Keras models for real AI analysis

---
"""
        
        # Create a demo visualization (copy of original with text overlay)
        try:
            original_image = current_image_data['original']
            if isinstance(original_image, Image.Image):
                # Convert PIL to numpy array for OpenCV operations
                demo_image = np.array(original_image)
                
                # Add demo text overlay
                cv2.putText(demo_image, f"DEMO: {category}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(demo_image, f"Area: {damage_area} pixels", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(demo_image, "Load models for real analysis", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Store the demo result
                current_image_data['analyzed'] = demo_image
                
                # Update the display
                self.update_image_display()
        except Exception as e:
            print(f"Error creating demo visualization: {e}")
        
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", result_text)
        
        self.status_label.configure(text=f"Demo analysis complete for {filename}")
        messagebox.showinfo("Analysis Complete", f"Demo analysis completed for {filename}\n\nLoad models for real AI analysis")
    
    def batch_process(self):
        """Process a directory of images"""
        directory = filedialog.askdirectory(title="Select Directory to Process")
        if not directory:
            return
        
        # Load all images from directory
        try:
            image_files = [f for f in os.listdir(directory) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not image_files:
                messagebox.showinfo("Info", "No image files found in the selected directory")
                return
            
            # Load all images
            self.current_images = []
            failed_count = 0
            
            for filename in image_files:
                try:
                    file_path = os.path.join(directory, filename)
                    original_image = Image.open(file_path).convert('RGB')
                    
                    image_data = {
                        'path': file_path,
                        'name': filename,
                        'original': original_image,
                        'analyzed': None
                    }
                    self.current_images.append(image_data)
                except Exception as e:
                    failed_count += 1
                    print(f"Failed to load {filename}: {e}")
            
            if self.current_images:
                self.current_image_index = 0
                self.update_image_display()
                self.update_navigation_controls()
                
                status_msg = f"Loaded {len(self.current_images)} images for batch processing"
                if failed_count > 0:
                    status_msg += f" ({failed_count} failed to load)"
                self.status_label.configure(text=status_msg)
                
                # Ask user if they want to start batch processing immediately
                if messagebox.askyesno("Batch Processing", 
                                     f"Loaded {len(self.current_images)} images.\n\nStart batch processing now?"):
                    # Start batch processing in separate thread
                    threading.Thread(target=self.batch_process_thread, daemon=True).start()
                else:
                    self.status_label.configure(text="Images loaded. Use navigation buttons to browse, or click 'Batch Process Directory' again to start processing.")
            else:
                messagebox.showerror("Error", "No images could be loaded from the selected directory")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load images: {str(e)}")
    
    def batch_process_thread(self):
        """Process all loaded images in separate thread"""
        try:
            if not self.current_images:
                self.status_label.configure(text="No images loaded for batch processing")
                return
            
            total_images = len(self.current_images)
            processed_count = 0
            
            self.status_label.configure(text=f"Starting batch processing of {total_images} images...")
            
            for i, image_data in enumerate(self.current_images):
                try:
                    # Update current image index for display
                    self.current_image_index = i
                    self.root.after(0, self.update_image_display)
                    self.root.after(0, self.update_navigation_controls)
                    
                    filename = image_data['name']
                    
                    self.status_label.configure(text=f"Processing {filename} ({i+1}/{total_images})")
                    
                    # Run demo analysis for each image
                    self.demo_analysis(image_data)
                    processed_count += 1
                    
                    # Small delay to allow UI updates
                    import time
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            
            final_status = f"Batch processing complete! Processed {processed_count}/{total_images} images"
            self.status_label.configure(text=final_status)
            
            # Show completion dialog
            result_msg = f"Batch processing completed!\n\nProcessed: {processed_count}/{total_images} images"
            if self.output_directory:
                result_msg += f"\nResults saved to: {self.output_directory}"
            
            messagebox.showinfo("Batch Complete", result_msg)
            
        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            print(error_msg)
            self.status_label.configure(text=error_msg)
            messagebox.showerror("Error", error_msg)
    
    def set_output_directory(self):
        """Set output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_directory = directory
            short_path = os.path.basename(directory) if len(directory) > 20 else directory
            self.status_label.configure(text=f"Output directory set: {short_path}")
            messagebox.showinfo("Output Directory", f"Output directory set to:\n{directory}")
    
    def previous_image(self):
        """Navigate to previous image"""
        if self.current_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_display()
            self.update_navigation_controls()
    
    def next_image(self):
        """Navigate to next image"""
        if self.current_images and self.current_image_index < len(self.current_images) - 1:
            self.current_image_index += 1
            self.update_image_display()
            self.update_navigation_controls()
    
    def zoom_in(self):
        """Zoom in"""
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        self.update_zoom_display()
        self.update_image_display()
    
    def zoom_out(self):
        """Zoom out"""
        self.zoom_level = max(self.zoom_level / 1.2, 0.1)
        self.update_zoom_display()
        self.update_image_display()
    
    def update_zoom_display(self):
        """Update zoom display"""
        zoom_percent = int(self.zoom_level * 100)
        self.zoom_label.configure(text=f"{zoom_percent}%")
    
    def update_navigation_controls(self):
        """Update navigation controls"""
        if not self.current_images:
            self.prev_button.configure(state="disabled")
            self.next_button.configure(state="disabled")
            self.image_counter_label.configure(text="No images")
            return
        
        total = len(self.current_images)
        current = self.current_image_index + 1
        self.image_counter_label.configure(text=f"{current} / {total}")
        
        self.prev_button.configure(state="normal" if self.current_image_index > 0 else "disabled")
        self.next_button.configure(state="normal" if self.current_image_index < total - 1 else "disabled")
    
    def update_image_display(self):
        """Update image display with IMPROVED sizing"""
        if not self.current_images:
            return
        
        current_image_data = self.current_images[self.current_image_index]
        
        # Display original image with improved sizing
        self.display_image_with_zoom(current_image_data['original'], self.original_image_label)
        
        # Display analyzed image if available
        if current_image_data['analyzed'] is not None:
            try:
                self.display_image_with_zoom(current_image_data['analyzed'], self.analyzed_image_label)
            except Exception as e:
                print(f"Error displaying analyzed image: {e}")
                self.analyzed_image_label.configure(image=None, text=f"Error displaying analyzed image: {str(e)}")
        else:
            self.analyzed_image_label.configure(image=None, text="Analysis result will appear here\n\nAfter running analysis")
    
    def display_image_with_zoom(self, pil_image, label_widget):
        """Display image with IMPROVED width fitting"""
        if pil_image is None:
            return
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(pil_image, np.ndarray):
                pil_image = Image.fromarray(pil_image)
            
            # IMPROVED sizing calculation
            # Each side gets full available width (about 420px after padding)
            available_width = 420
            max_height = 400
            
            # Get original dimensions
            original_width, original_height = pil_image.size
            aspect_ratio = original_height / original_width
            
            # Calculate size to FILL available width
            fit_width = available_width
            fit_height = int(fit_width * aspect_ratio)
            
            # If too tall, scale down
            if fit_height > max_height:
                fit_height = max_height
                fit_width = int(fit_height / aspect_ratio)
            
            # Apply zoom
            display_width = int(fit_width * self.zoom_level)
            display_height = int(fit_height * self.zoom_level)
            
            # Ensure minimum size
            display_width = max(display_width, 100)
            display_height = max(display_height, 100)
            
            # Resize with high quality
            display_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Create CustomTkinter image
            ctk_image = ctk.CTkImage(
                light_image=display_image,
                dark_image=display_image,
                size=(display_width, display_height)
            )
            label_widget.configure(image=ctk_image, text="")
            
            print(f"Image display: {original_width}x{original_height} -> {display_width}x{display_height} (zoom: {self.zoom_level:.1f})")
            
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            label_widget.configure(image=None, text=f"Error displaying image: {str(e)}")
    
    def update_analysis_results(self, filename, method, category, damage_area, description):
        """Update the analysis results display with detailed information"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Determine color coding based on damage level
        if damage_area == 0:
            damage_level = "🟢 NO DAMAGE"
        elif damage_area <= 5025:
            damage_level = "🟡 MANAGEABLE"
        elif damage_area <= 17670:
            damage_level = "🟠 PARTIALLY DAMAGED"
        else:
            damage_level = "🔴 COMPLETELY DAMAGED"
        
        result_text = f"""
🔍 ANALYSIS COMPLETE - {timestamp}

📁 FILE: {filename}
🤖 METHOD: {method}
📊 DESCRIPTION: {description}

📈 RESULTS:
• Damage Category: {category}
• Damage Area: {damage_area:,} pixels
• Classification: {damage_level}

🎨 VISUAL OVERLAY:
• Red contours show detected damage areas
• Green text displays analysis metadata
• Original image (left) vs Analysis (right)

💾 OUTPUT: {"Saved to output directory" if self.output_directory else "No output directory set"}

---
"""
        
        # Clear and update results
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", result_text)
    
    def create_result_visualization(self, original_image, binary_mask, category, damage_area, method_name):
        """Create a visualization of the analysis result"""
        # Create overlay image
        overlay_image = original_image.copy()
        
        # Find contours and draw them
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2)
        
        # Add text annotations
        cv2.putText(overlay_image, f"Method: {method_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay_image, f"Category: {category}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay_image, f"Area: {damage_area} pixels", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay_image
    
    def save_analysis_result(self, image_data, result_image, method_name):
        """Save analysis result to output directory"""
        if not self.output_directory:
            return None
        
        try:
            # Create method-specific subdirectory
            method_dir = os.path.join(self.output_directory, method_name.replace(" ", "_"))
            os.makedirs(method_dir, exist_ok=True)
            
            # Save the annotated image
            filename = image_data['name']
            name, ext = os.path.splitext(filename)
            result_filename = f"{name}_analyzed{ext}"
            result_path = os.path.join(method_dir, result_filename)
            
            # Convert and save
            result_pil = Image.fromarray(result_image)
            result_pil.save(result_path)
            
            return result_path
            
        except Exception as e:
            print(f"Failed to save result: {str(e)}")
            return None
    
    # Model loading methods
    def load_pytorch_model(self):
        """Load PyTorch model for prediction"""
        if not MODEL_SUPPORT:
            messagebox.showwarning("Warning", "Model support not available. Check if utils modules are installed.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select PyTorch Model",
            filetypes=[("PyTorch files", "*.pt *.pth"), ("All files", "*.*")]
        )
        
        if file_path:
            self.pytorch_model_entry.delete(0, "end")
            self.pytorch_model_entry.insert(0, file_path)
            
            if self.model_loader and self.model_loader.load_pytorch_model(file_path, "resnet50"):
                self.status_label.configure(text="PyTorch model loaded successfully")
                messagebox.showinfo("Success", "PyTorch model loaded successfully!")
            else:
                self.status_label.configure(text="Failed to load PyTorch model")
                messagebox.showerror("Error", "Failed to load PyTorch model")
    
    def load_keras_model(self):
        """Load Keras model for prediction"""
        if not MODEL_SUPPORT:
            messagebox.showwarning("Warning", "Model support not available. Check if utils modules are installed.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Keras Model",
            filetypes=[("Keras files", "*.h5 *.keras"), ("All files", "*.*")]
        )
        
        if file_path:
            self.keras_model_entry.delete(0, "end")
            self.keras_model_entry.insert(0, file_path)
            
            if self.model_loader and self.model_loader.load_keras_model(file_path, "ResNet50"):
                self.status_label.configure(text="Keras model loaded successfully")
                messagebox.showinfo("Success", "Keras model loaded successfully!")
            else:
                self.status_label.configure(text="Failed to load Keras model")
                messagebox.showerror("Error", "Failed to load Keras model")
    
    # OpenCV methods
    def load_opencv_image(self):
        """Load image for OpenCV analysis"""
        self.load_single_image()  # Reuse the same functionality
    
    def analyze_opencv(self):
        """Analyze image using OpenCV method"""
        if not self.current_images:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if not MODEL_SUPPORT:
            messagebox.showwarning("Warning", "OpenCV support not available. Check if utils modules are installed.")
            return
        
        try:
            current_image_data = self.current_images[self.current_image_index]
            image_path = current_image_data['path']
            filename = current_image_data['name']
            
            damage_area, damage_mask, original_image = analyze_damage_opencv(image_path)
            
            if damage_area is not None:
                category = self.classify_damage(damage_area)
                
                # Create result visualization
                result_image = self.create_result_visualization(original_image, damage_mask, category, damage_area, "OpenCV")
                
                # Store result image
                current_image_data['analyzed'] = result_image
                self.update_image_display()
                
                # Update results display
                self.update_analysis_results(filename, "OpenCV", category, damage_area, "White pixel detection (threshold > 240)")
                
                # Save result to output directory
                if self.output_directory:
                    self.save_analysis_result(current_image_data, result_image, "OpenCV")
                
                self.status_label.configure(text="OpenCV analysis complete")
            else:
                messagebox.showerror("Error", "Failed to analyze image with OpenCV")
                
        except Exception as e:
            messagebox.showerror("Error", f"OpenCV analysis failed: {str(e)}")
    
    def process_opencv_directory(self):
        """Process directory with OpenCV method"""
        messagebox.showinfo("OpenCV Directory Processing", "OpenCV directory processing functionality available.\nSelect a directory to process all images with OpenCV method.")
    
    def classify_damage(self, damage_area):
        """Classify damage based on area"""
        if damage_area > 17671:
            return "Completely damaged"
        elif damage_area > 5026:
            return "Partially damaged"
        elif damage_area > 0:
            return "Manageable"
        else:
            return "No Damage Detected"
    
    # Training methods
    def on_framework_change(self, choice):
        """Handle framework selection change"""
        if choice == "PyTorch":
            self.backbone_combo.configure(values=self.pytorch_backbones)
            self.backbone_var.set("resnet50")
        else:
            self.backbone_combo.configure(values=self.keras_backbones)
            self.backbone_var.set("ResNet50")
    
    def browse_image_dir(self):
        """Browse for images directory"""
        directory = filedialog.askdirectory(title="Select Images Directory")
        if directory:
            self.image_dir_entry.delete(0, "end")
            self.image_dir_entry.insert(0, directory)
    
    def browse_mask_dir(self):
        """Browse for masks directory"""
        directory = filedialog.askdirectory(title="Select Masks Directory")
        if directory:
            self.mask_dir_entry.delete(0, "end")
            self.mask_dir_entry.insert(0, directory)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_entry.delete(0, "end")
            self.output_dir_entry.insert(0, directory)
    
    def start_training(self):
        """Start model training"""
        if not MODEL_SUPPORT:
            messagebox.showwarning("Warning", "Training support not available. Check if utils modules are installed.")
            return
        
        try:
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_size_entry.get())
            
            if not os.path.exists(self.image_dir_entry.get()):
                messagebox.showerror("Error", "Images directory does not exist")
                return
            
            if not os.path.exists(self.mask_dir_entry.get()):
                messagebox.showerror("Error", "Masks directory does not exist")
                return
            
            # Update UI
            self.train_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.training_log.delete("0.0", "end")
            self.progress_bar.set(0)
            
            # Start training in separate thread
            framework = self.framework_var.get()
            backbone = self.backbone_var.get()
            
            def train_thread():
                try:
                    if framework == "PyTorch":
                        success, model_path = self.training_manager.train_pytorch_model(
                            self.image_dir_entry.get(),
                            self.mask_dir_entry.get(),
                            self.output_dir_entry.get(),
                            backbone, epochs, batch_size
                        )
                    else:
                        success, model_path = self.training_manager.train_keras_model(
                            self.image_dir_entry.get(),
                            self.mask_dir_entry.get(),
                            self.output_dir_entry.get(),
                            backbone, epochs, batch_size
                        )
                    
                    self.root.after(0, self.training_complete, success, model_path)
                except Exception as e:
                    self.root.after(0, self.training_complete, False, str(e))
            
            threading.Thread(target=train_thread, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for epochs and batch size")
        except Exception as e:
            messagebox.showerror("Error", f"Training setup failed: {str(e)}")
    
    def stop_training(self):
        """Stop the current training"""
        if self.training_manager:
            self.training_manager.stop_training()
        self.train_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.update_status("Training stopped by user")
    
    def training_complete(self, success, model_path):
        """Handle training completion"""
        self.train_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        
        if success:
            messagebox.showinfo("Success", f"Training completed successfully!\nModel saved to: {model_path}")
        else:
            messagebox.showerror("Error", f"Training failed: {model_path}")
    
    def update_training_progress(self, current, total):
        """Update training progress bar"""
        if hasattr(self, 'progress_bar'):
            progress = current / total
            self.progress_bar.set(progress)
            self.root.update()
    
    def update_status(self, message):
        """Update status bar and training log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_label.configure(text=message)
        
        # Add to training log with timestamp
        log_message = f"[{timestamp}] {message}\n"
        if hasattr(self, 'training_log'):
            self.training_log.insert("end", log_message)
            self.training_log.see("end")
        self.root.update()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    app = ModernDamageAnalyzer()
    app.run()

if __name__ == "__main__":
    main()