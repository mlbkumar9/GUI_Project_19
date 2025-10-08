#!/usr/bin/env python3
"""
Structural Damage Analyzer - Clean and Readable Version
A modern GUI application for training and analyzing structural damage using AI models.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
import datetime

from utils.model_loader import ModelLoader
from utils.image_utils import load_and_preprocess_image, analyze_damage_opencv, save_results
from utils.training_utils import TrainingManager
from utils.ui_theme import SimpleTheme, SimpleButton, SimpleText, create_status_bar, DARK_COLORS
from assets.logo import create_app_logo


class StructuralDamageAnalyzer:
    """Main application class for the Structural Damage Analyzer"""
    
    def __init__(self, root):
        self.root = root
        self._initialize_window()
        self._initialize_components()
        self._initialize_variables()
        self.setup_ui()
    
    def _initialize_window(self):
        """Initialize the main window properties"""
        self.root.title("Structural Damage Analyzer - AI Detection & Training")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Configure window for dark theme
        self.root.configure(bg=DARK_COLORS['bg'])
        
        # Set app icon
        try:
            logo = create_app_logo(size=(32, 32))
            self.logo_photo = ImageTk.PhotoImage(logo)
            self.root.iconphoto(True, self.logo_photo)
        except Exception:
            pass  # Fallback if logo creation fails
    
    def _initialize_components(self):
        """Initialize core components"""
        self.theme = SimpleTheme(self.root)
        self.model_loader = ModelLoader()
        self.training_manager = TrainingManager(
            progress_callback=self.update_training_progress,
            status_callback=self.update_status
        )
    
    def _initialize_variables(self):
        """Initialize application variables"""
        self.current_image_path = None
        self.current_image = None
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
    
    def setup_ui(self):
        """Setup the main user interface"""
        self._create_header()
        self._create_main_notebook()
        self._create_status_bar()
    
    def _create_header(self):
        """Create the application header with logo and title"""
        header_frame = tk.Frame(self.root, bg=DARK_COLORS['bg'], height=80)
        header_frame.pack(fill=tk.X, padx=15, pady=(15, 5))
        header_frame.pack_propagate(False)
        
        # Logo
        try:
            logo = create_app_logo(size=(64, 64))
            self.header_logo = ImageTk.PhotoImage(logo)
            logo_label = tk.Label(header_frame, image=self.header_logo, bg=DARK_COLORS['bg'])
            logo_label.pack(side=tk.LEFT, padx=(10, 20), pady=10)
        except Exception:
            pass
        
        # Title section
        title_frame = tk.Frame(header_frame, bg=DARK_COLORS['bg'])
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="Structural Damage Analyzer",
            font=('Segoe UI', 20, 'bold'),
            fg=DARK_COLORS['fg'],
            bg=DARK_COLORS['bg']
        )
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(
            title_frame,
            text="AI-Powered Damage Detection & Model Training Suite",
            font=('Segoe UI', 12),
            fg=DARK_COLORS['accent'],
            bg=DARK_COLORS['bg']
        )
        subtitle_label.pack(anchor='w', pady=(5, 0))
    
    def _create_main_notebook(self):
        """Create the main tabbed interface"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=(10, 15))
        
        # Create tabs
        self.training_frame = ttk.Frame(notebook)
        self.prediction_frame = ttk.Frame(notebook)
        self.opencv_frame = ttk.Frame(notebook)
        
        notebook.add(self.training_frame, text="🚀 Model Training")
        notebook.add(self.prediction_frame, text="🔍 Damage Prediction")
        notebook.add(self.opencv_frame, text="📊 OpenCV Analysis")
        
        # Setup tab content
        self._setup_training_tab()
        self._setup_prediction_tab()
        self._setup_opencv_tab()
    
    def _create_status_bar(self):
        """Create the status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Welcome to Structural Damage Analyzer")
        self.status_bar = create_status_bar(self.root, self.status_var)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Training Tab Methods
    def _setup_training_tab(self):
        """Setup the training interface"""
        main_frame = ttk.Frame(self.training_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self._create_training_config_section(main_frame)
        self._create_directory_selection_section(main_frame)
        self._create_training_controls_section(main_frame)
        self._create_training_log_section(main_frame)
    
    def _create_training_config_section(self, parent):
        """Create training configuration section"""
        config_frame = ttk.LabelFrame(parent, text="Training Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Framework selection
        ttk.Label(config_frame, text="Framework:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.framework_var = tk.StringVar(value="PyTorch")
        framework_combo = ttk.Combobox(
            config_frame, textvariable=self.framework_var,
            values=["PyTorch", "Keras"], state="readonly", width=15
        )
        framework_combo.grid(row=0, column=1, padx=(0, 20))
        framework_combo.bind('<<ComboboxSelected>>', self._on_framework_change)
        
        # Backbone selection
        ttk.Label(config_frame, text="Backbone:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.backbone_var = tk.StringVar(value="resnet50")
        self.backbone_combo = ttk.Combobox(
            config_frame, textvariable=self.backbone_var,
            values=self.pytorch_backbones, state="readonly", width=15
        )
        self.backbone_combo.grid(row=0, column=3, padx=(0, 20))
        
        # Training parameters
        ttk.Label(config_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        self.epochs_var = tk.StringVar(value="25")
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=10).grid(row=1, column=1, pady=(10, 0))
        
        ttk.Label(config_frame, text="Batch Size:").grid(row=1, column=2, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        self.batch_size_var = tk.StringVar(value="4")
        ttk.Entry(config_frame, textvariable=self.batch_size_var, width=10).grid(row=1, column=3, pady=(10, 0))
    
    def _create_directory_selection_section(self, parent):
        """Create directory selection section"""
        dir_frame = ttk.LabelFrame(parent, text="Data Directories", padding="10")
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Images directory
        ttk.Label(dir_frame, text="Images Directory:").grid(row=0, column=0, sticky=tk.W)
        self.image_dir_var = tk.StringVar(value=os.path.join(self.base_dir, "RAW_Images"))
        ttk.Entry(dir_frame, textvariable=self.image_dir_var, width=50).grid(row=0, column=1, padx=(5, 5))
        SimpleButton(dir_frame, text="Browse", command=self._browse_image_dir, style='info').grid(row=0, column=2)
        
        # Masks directory
        ttk.Label(dir_frame, text="Masks Directory:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.mask_dir_var = tk.StringVar(value=os.path.join(self.base_dir, "Masks"))
        ttk.Entry(dir_frame, textvariable=self.mask_dir_var, width=50).grid(row=1, column=1, padx=(5, 5), pady=(5, 0))
        SimpleButton(dir_frame, text="Browse", command=self._browse_mask_dir, style='info').grid(row=1, column=2, pady=(5, 0))
        
        # Output directory
        ttk.Label(dir_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.output_dir_var = tk.StringVar(value=os.path.join(self.base_dir, "Trained_Models"))
        ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=50).grid(row=2, column=1, padx=(5, 5), pady=(5, 0))
        SimpleButton(dir_frame, text="Browse", command=self._browse_output_dir, style='info').grid(row=2, column=2, pady=(5, 0))
    
    def _create_training_controls_section(self, parent):
        """Create training controls section"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.train_button = SimpleButton(
            control_frame, text="Start Training",
            command=self.start_training, style='success'
        )
        self.train_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = SimpleButton(
            control_frame, text="Stop Training",
            command=self.stop_training, style='danger', state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
    
    def _create_training_log_section(self, parent):
        """Create training log section"""
        log_frame = ttk.LabelFrame(parent, text="Training Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.training_log = SimpleText(log_frame, height=15)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.training_log.yview)
        self.training_log.configure(yscrollcommand=log_scrollbar.set)
        
        self.training_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Prediction Tab Methods
    def _setup_prediction_tab(self):
        """Setup the prediction interface"""
        main_frame = ttk.Frame(self.prediction_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self._create_model_loading_section(main_frame)
        self._create_image_analysis_section(main_frame)
        self._create_results_display_section(main_frame)
    
    def _create_model_loading_section(self, parent):
        """Create model loading section"""
        model_frame = ttk.LabelFrame(parent, text="Model Loading", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # PyTorch model
        ttk.Label(model_frame, text="PyTorch Model:").grid(row=0, column=0, sticky=tk.W)
        self.pytorch_model_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.pytorch_model_var, width=40).grid(row=0, column=1, padx=(5, 5))
        SimpleButton(model_frame, text="Load", command=self.load_pytorch_model, style='primary').grid(row=0, column=2)
        
        ttk.Label(model_frame, text="Backbone:").grid(row=0, column=3, sticky=tk.W, padx=(20, 5))
        self.pytorch_backbone_var = tk.StringVar(value="resnet50")
        ttk.Combobox(
            model_frame, textvariable=self.pytorch_backbone_var,
            values=self.pytorch_backbones, state="readonly", width=15
        ).grid(row=0, column=4)
        
        # Keras model
        ttk.Label(model_frame, text="Keras Model:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.keras_model_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.keras_model_var, width=40).grid(row=1, column=1, padx=(5, 5), pady=(10, 0))
        SimpleButton(model_frame, text="Load", command=self.load_keras_model, style='primary').grid(row=1, column=2, pady=(10, 0))
        
        ttk.Label(model_frame, text="Backbone:").grid(row=1, column=3, sticky=tk.W, padx=(20, 5), pady=(10, 0))
        self.keras_backbone_var = tk.StringVar(value="ResNet50")
        ttk.Combobox(
            model_frame, textvariable=self.keras_backbone_var,
            values=self.keras_backbones, state="readonly", width=15
        ).grid(row=1, column=4, pady=(10, 0))
    
    def _create_image_analysis_section(self, parent):
        """Create image analysis section"""
        analysis_frame = ttk.LabelFrame(parent, text="Image Analysis", padding="10")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        SimpleButton(analysis_frame, text="Load Single Image", command=self.load_single_image, style='info').pack(side=tk.LEFT, padx=(0, 10))
        SimpleButton(analysis_frame, text="Batch Process Directory", command=self.batch_process, style='warning').pack(side=tk.LEFT, padx=(0, 10))
        SimpleButton(analysis_frame, text="Analyze Current Image", command=self.analyze_current_image, style='success').pack(side=tk.LEFT)
    
    def _create_results_display_section(self, parent):
        """Create results display section"""
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill=tk.BOTH, expand=True)
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Image display
        image_frame = ttk.LabelFrame(results_frame, text="Current Image", padding="5")
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.image_label = ttk.Label(image_frame, text="No image loaded\n\nClick 'Load Single Image' to get started")
        self.image_label.pack(expand=True)
        
        # Results display
        results_text_frame = ttk.LabelFrame(results_frame, text="Analysis Results", padding="5")
        results_text_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.results_text = SimpleText(results_text_frame, width=40, height=20)
        results_scrollbar = ttk.Scrollbar(results_text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        # Add initial helpful content
        initial_content = self._get_initial_results_content()
        self.results_text.insert(tk.END, initial_content)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # OpenCV Tab Methods
    def _setup_opencv_tab(self):
        """Setup the OpenCV analysis tab"""
        main_frame = ttk.Frame(self.opencv_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self._create_opencv_instructions(main_frame)
        self._create_opencv_controls(main_frame)
        self._create_opencv_results(main_frame)
    
    def _create_opencv_instructions(self, parent):
        """Create OpenCV instructions section"""
        instructions_frame = tk.Frame(parent, bg=DARK_COLORS['button_bg'], relief='solid', bd=1)
        instructions_frame.pack(fill=tk.X, pady=(0, 15))
        
        instructions_text = (
            "OpenCV Analysis: Traditional computer vision method that detects damage "
            "based on white pixels (threshold > 240)\n"
            "This method works well for images with visible white/bright damage areas "
            "and doesn't require trained models."
        )
        
        instructions = tk.Label(
            instructions_frame,
            text=instructions_text,
            font=('Segoe UI', 11),
            fg=DARK_COLORS['fg'],
            bg=DARK_COLORS['button_bg'],
            justify=tk.LEFT,
            wraplength=800
        )
        instructions.pack(padx=15, pady=10)
    
    def _create_opencv_controls(self, parent):
        """Create OpenCV controls section"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        SimpleButton(control_frame, text="Load Image", command=self.load_opencv_image, style='info').pack(side=tk.LEFT, padx=(0, 10))
        SimpleButton(control_frame, text="Analyze with OpenCV", command=self.analyze_opencv, style='success').pack(side=tk.LEFT, padx=(0, 10))
        SimpleButton(control_frame, text="Process Directory", command=self.process_opencv_directory, style='warning').pack(side=tk.LEFT)
    
    def _create_opencv_results(self, parent):
        """Create OpenCV results section"""
        opencv_results_frame = ttk.Frame(parent)
        opencv_results_frame.pack(fill=tk.BOTH, expand=True)
        opencv_results_frame.columnconfigure(0, weight=1)
        opencv_results_frame.columnconfigure(1, weight=1)
        opencv_results_frame.rowconfigure(0, weight=1)
        
        # Image display
        opencv_image_frame = ttk.LabelFrame(opencv_results_frame, text="Original Image", padding="5")
        opencv_image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.opencv_image_label = ttk.Label(opencv_image_frame, text="No image loaded\n\nClick 'Load Image' to get started")
        self.opencv_image_label.pack(expand=True)
        
        # Results
        opencv_text_frame = ttk.LabelFrame(opencv_results_frame, text="OpenCV Results", padding="5")
        opencv_text_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.opencv_results_text = SimpleText(opencv_text_frame, width=40, height=20)
        opencv_scrollbar = ttk.Scrollbar(opencv_text_frame, orient=tk.VERTICAL, command=self.opencv_results_text.yview)
        self.opencv_results_text.configure(yscrollcommand=opencv_scrollbar.set)
        
        # Add initial content
        opencv_initial_content = self._get_initial_opencv_content()
        self.opencv_results_text.insert(tk.END, opencv_initial_content)
        
        self.opencv_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        opencv_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Helper Methods
    def _get_initial_results_content(self):
        """Get initial content for results text area"""
        return """AI Damage Analysis Results

Instructions:
1. Load a trained model (PyTorch or Keras)
2. Load an image to analyze
3. Click 'Analyze Current Image'

The AI will detect and classify damage as:
• No Damage (0 pixels)
• Manageable (1-5,025 pixels)
• Partially Damaged (5,026-17,670 pixels)
• Completely Damaged (>17,671 pixels)

Results will appear here after analysis...
"""
    
    def _get_initial_opencv_content(self):
        """Get initial content for OpenCV results text area"""
        return """OpenCV Analysis Results

Instructions:
1. Load an image with visible damage
2. Click 'Analyze with OpenCV'

How it works:
• Detects white/bright pixels (threshold > 240)
• Counts damage area in pixels
• No AI models required
• Good for baseline comparison

Damage Classification:
• No Damage (0 pixels)
• Manageable (1-5,025 pixels)
• Partially Damaged (5,026-17,670 pixels)
• Completely Damaged (>17,671 pixels)

Results will appear here after analysis...
"""
    
    # Event Handlers
    def _on_framework_change(self, event=None):
        """Handle framework selection change"""
        framework = self.framework_var.get()
        if framework == "PyTorch":
            self.backbone_combo.configure(values=self.pytorch_backbones)
            self.backbone_var.set("resnet50")
        else:
            self.backbone_combo.configure(values=self.keras_backbones)
            self.backbone_var.set("ResNet50")
    
    def _browse_image_dir(self):
        """Browse for images directory"""
        directory = filedialog.askdirectory(title="Select Images Directory")
        if directory:
            self.image_dir_var.set(directory)
    
    def _browse_mask_dir(self):
        """Browse for masks directory"""
        directory = filedialog.askdirectory(title="Select Masks Directory")
        if directory:
            self.mask_dir_var.set(directory)
    
    def _browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def update_status(self, message):
        """Update status bar and training log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        status_message = f"{message}"
        self.status_var.set(status_message)
        
        # Add to training log with timestamp
        log_message = f"[{timestamp}] {message}\n"
        if hasattr(self, 'training_log'):
            self.training_log.insert(tk.END, log_message)
            self.training_log.see(tk.END)
        self.root.update()
    
    def update_training_progress(self, current, total):
        """Update training progress bar"""
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.root.update()
    
    # Training Methods
    def start_training(self):
        """Start model training"""
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            
            if not os.path.exists(self.image_dir_var.get()):
                messagebox.showerror("Error", "Images directory does not exist")
                return
            
            if not os.path.exists(self.mask_dir_var.get()):
                messagebox.showerror("Error", "Masks directory does not exist")
                return
            
            self._start_training_thread(epochs, batch_size)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for epochs and batch size")
    
    def _start_training_thread(self, epochs, batch_size):
        """Start training in a separate thread"""
        # Update UI
        self.train_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.training_log.delete(1.0, tk.END)
        self.progress_var.set(0)
        
        framework = self.framework_var.get()
        backbone = self.backbone_var.get()
        
        def train_thread():
            if framework == "PyTorch":
                success, model_path = self.training_manager.train_pytorch_model(
                    self.image_dir_var.get(),
                    self.mask_dir_var.get(),
                    self.output_dir_var.get(),
                    backbone, epochs, batch_size
                )
            else:
                success, model_path = self.training_manager.train_keras_model(
                    self.image_dir_var.get(),
                    self.mask_dir_var.get(),
                    self.output_dir_var.get(),
                    backbone, epochs, batch_size
                )
            
            self.root.after(0, self._training_complete, success, model_path)
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def stop_training(self):
        """Stop the current training"""
        self.training_manager.stop_training()
        self.train_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.update_status("Training stopped by user")
    
    def _training_complete(self, success, model_path):
        """Handle training completion"""
        self.train_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        
        if success:
            messagebox.showinfo("Success", f"Training completed successfully!\nModel saved to: {model_path}")
        else:
            messagebox.showerror("Error", "Training failed. Check the log for details.")
    
    # Model Loading Methods
    def load_pytorch_model(self):
        """Load PyTorch model for prediction"""
        file_path = filedialog.askopenfilename(
            title="Select PyTorch Model",
            filetypes=[("PyTorch files", "*.pt *.pth"), ("All files", "*.*")]
        )
        
        if file_path:
            self.pytorch_model_var.set(file_path)
            backbone = self.pytorch_backbone_var.get()
            
            if self.model_loader.load_pytorch_model(file_path, backbone):
                self.update_status("PyTorch model loaded successfully")
                messagebox.showinfo("Success", "PyTorch model loaded successfully!")
            else:
                self.update_status("Failed to load PyTorch model")
                messagebox.showerror("Error", "Failed to load PyTorch model")
    
    def load_keras_model(self):
        """Load Keras model for prediction"""
        file_path = filedialog.askopenfilename(
            title="Select Keras Model",
            filetypes=[("Keras files", "*.h5 *.keras"), ("All files", "*.*")]
        )
        
        if file_path:
            self.keras_model_var.set(file_path)
            backbone = self.keras_backbone_var.get()
            
            if self.model_loader.load_keras_model(file_path, backbone):
                self.update_status("Keras model loaded successfully")
                messagebox.showinfo("Success", "Keras model loaded successfully!")
            else:
                self.update_status("Failed to load Keras model")
                messagebox.showerror("Error", "Failed to load Keras model")
    
    # Image Analysis Methods
    def load_single_image(self):
        """Load single image for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.current_image = load_and_preprocess_image(file_path, target_size=(400, 400))
                
                # Display image
                display_image = self.current_image.copy()
                display_image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(display_image)
                
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                self.update_status(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def analyze_current_image(self):
        """Analyze the currently loaded image"""
        if self.current_image_path is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.model_loader.pytorch_model is None and self.model_loader.keras_model is None:
            messagebox.showwarning("Warning", "Please load at least one model first")
            return
        
        try:
            self.update_status("Analyzing image with AI models...")
            results = self._perform_ai_analysis()
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "\n".join(results))
            
            self.update_status("Analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def _perform_ai_analysis(self):
        """Perform AI analysis on current image"""
        results = []
        
        # PyTorch prediction
        if self.model_loader.pytorch_model is not None:
            try:
                binary_mask, original_image = self.model_loader.predict_pytorch_segmentation(self.current_image_path)
                category, damage_area = self.model_loader.analyze_damage(binary_mask)
                
                results.extend([
                    "PyTorch U-Net Results:",
                    "-" * 25,
                    f"Category: {category}",
                    f"Damage Area: {damage_area} pixels",
                    ""
                ])
            except Exception as e:
                results.append(f"PyTorch prediction error: {str(e)}\n")
        
        # Keras prediction
        if self.model_loader.keras_model is not None:
            try:
                binary_mask, original_image = self.model_loader.predict_keras_segmentation(self.current_image_path)
                category, damage_area = self.model_loader.analyze_damage(binary_mask)
                
                results.extend([
                    "Keras U-Net++ Results:",
                    "-" * 25,
                    f"Category: {category}",
                    f"Damage Area: {damage_area} pixels"
                ])
            except Exception as e:
                results.append(f"Keras prediction error: {str(e)}")
        
        return results
    
    def batch_process(self):
        """Process a directory of images"""
        directory = filedialog.askdirectory(title="Select Directory to Process")
        if not directory:
            return
        
        if self.model_loader.pytorch_model is None and self.model_loader.keras_model is None:
            messagebox.showwarning("Warning", "Please load at least one model first")
            return
        
        threading.Thread(target=self._batch_process_thread, args=(directory,), daemon=True).start()
    
    def _batch_process_thread(self, directory):
        """Process directory in separate thread"""
        try:
            image_files = [f for f in os.listdir(directory) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not image_files:
                messagebox.showinfo("Info", "No image files found in the selected directory")
                return
            
            output_dir = os.path.join(directory, "Analysis_Results")
            os.makedirs(output_dir, exist_ok=True)
            
            results_summary = []
            
            for i, filename in enumerate(image_files):
                image_path = os.path.join(directory, filename)
                self.update_status(f"Processing {filename} ({i+1}/{len(image_files)})")
                
                try:
                    if self.model_loader.pytorch_model is not None:
                        binary_mask, original_image = self.model_loader.predict_pytorch_segmentation(image_path)
                        category, damage_area = self.model_loader.analyze_damage(binary_mask)
                        
                        save_results(original_image, binary_mask, category, damage_area, filename, 
                                   os.path.join(output_dir, "PyTorch"))
                        
                        results_summary.append(f"{filename}: {category} ({damage_area} pixels)")
                
                except Exception as e:
                    results_summary.append(f"{filename}: Error - {str(e)}")
            
            # Save summary
            with open(os.path.join(output_dir, "batch_results.txt"), 'w') as f:
                f.write("\n".join(results_summary))
            
            self.update_status(f"Batch processing complete. Results saved to {output_dir}")
            messagebox.showinfo("Complete", f"Processed {len(image_files)} images.\nResults saved to: {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")
    
    # OpenCV Methods
    def load_opencv_image(self):
        """Load image for OpenCV analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Image for OpenCV Analysis",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.opencv_image_path = file_path
                display_image = load_and_preprocess_image(file_path, target_size=(400, 400))
                
                photo = ImageTk.PhotoImage(display_image)
                self.opencv_image_label.configure(image=photo, text="")
                self.opencv_image_label.image = photo
                
                self.update_status(f"OpenCV image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def analyze_opencv(self):
        """Analyze image using OpenCV method"""
        if not hasattr(self, 'opencv_image_path'):
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            damage_area, damage_mask, original_image = analyze_damage_opencv(self.opencv_image_path)
            
            if damage_area is not None:
                category = self._classify_damage(damage_area)
                
                results = [
                    "OpenCV Analysis Results:",
                    "-" * 25,
                    f"Category: {category}",
                    f"Damage Area: {damage_area} pixels",
                    f"Method: White pixel detection (threshold > 240)",
                    "",
                    "This method detects existing white/bright damage",
                    "areas in the image without using ML models."
                ]
                
                self.opencv_results_text.delete(1.0, tk.END)
                self.opencv_results_text.insert(tk.END, "\n".join(results))
                
                self.update_status("OpenCV analysis complete")
            else:
                messagebox.showerror("Error", "Failed to analyze image with OpenCV")
                
        except Exception as e:
            messagebox.showerror("Error", f"OpenCV analysis failed: {str(e)}")
    
    def process_opencv_directory(self):
        """Process directory with OpenCV method"""
        directory = filedialog.askdirectory(title="Select Directory for OpenCV Processing")
        if not directory:
            return
        
        threading.Thread(target=self._opencv_process_thread, args=(directory,), daemon=True).start()
    
    def _opencv_process_thread(self, directory):
        """Process directory with OpenCV in separate thread"""
        try:
            image_files = [f for f in os.listdir(directory) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not image_files:
                messagebox.showinfo("Info", "No image files found in the selected directory")
                return
            
            output_dir = os.path.join(directory, "OpenCV_Analysis")
            os.makedirs(output_dir, exist_ok=True)
            
            results_summary = []
            
            for i, filename in enumerate(image_files):
                image_path = os.path.join(directory, filename)
                self.update_status(f"OpenCV processing {filename} ({i+1}/{len(image_files)})")
                
                damage_area, damage_mask, original_image = analyze_damage_opencv(image_path)
                
                if damage_area is not None:
                    category = self._classify_damage(damage_area)
                    
                    save_results(original_image, damage_mask, category, damage_area, filename, output_dir)
                    results_summary.append(f"{filename}: {category} ({damage_area} pixels)")
                else:
                    results_summary.append(f"{filename}: Processing failed")
            
            # Save summary
            with open(os.path.join(output_dir, "opencv_results.txt"), 'w') as f:
                f.write("OpenCV Analysis Results (White Pixel Detection)\n")
                f.write("=" * 50 + "\n")
                f.write("\n".join(results_summary))
            
            self.update_status(f"OpenCV batch processing complete. Results saved to {output_dir}")
            messagebox.showinfo("Complete", f"Processed {len(image_files)} images with OpenCV.\nResults saved to: {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"OpenCV batch processing failed: {str(e)}")
    
    def _classify_damage(self, damage_area):
        """Classify damage based on area"""
        MANAGEABLE_AREA_THRESHOLD = 5026
        PARTIALLY_DAMAGED_AREA_THRESHOLD = 17671
        
        if damage_area > PARTIALLY_DAMAGED_AREA_THRESHOLD:
            return "Completely damaged"
        elif damage_area > MANAGEABLE_AREA_THRESHOLD:
            return "Partially damaged"
        elif damage_area > 0:
            return "Manageable"
        else:
            return "No Damage Detected"


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = StructuralDamageAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()