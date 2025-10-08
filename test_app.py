#!/usr/bin/env python3
"""
Test version of the Structural Damage Analyzer
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class TestDamageAnalyzer:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Structural Damage Analyzer - Test Version")
        self.root.geometry("1400x900")
        
        # Variables
        self.current_images = []
        self.current_image_index = 0
        self.zoom_level = 1.0
        self.output_directory = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create header
        header_frame = ctk.CTkFrame(self.root, height=70)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(header_frame, text="🏗️ Structural Damage Analyzer - Test", 
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
        self.status_label = ctk.CTkLabel(self.root, text="Ready - Test Version", 
                                        font=ctk.CTkFont(size=12))
        self.status_label.pack(side="bottom", fill="x", padx=20, pady=(0, 10))
    
    def setup_controls(self):
        """Setup control buttons"""
        # Create tabview
        self.tabview = ctk.CTkTabview(self.left_sidebar, width=430)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add tabs
        self.tabview.add("🔍 Analysis")
        self.tabview.add("📊 Results")
        
        # Analysis tab
        analysis_tab = self.tabview.tab("🔍 Analysis")
        
        # Buttons with FIXED text
        ctk.CTkButton(analysis_tab, text="📷 Load Single Image", 
                     command=self.load_single_image, height=35, width=200).pack(pady=5)
        ctk.CTkButton(analysis_tab, text="📷 Load Multiple Images", 
                     command=self.load_multiple_images, height=35, width=200).pack(pady=5)
        ctk.CTkButton(analysis_tab, text="🔍 Analyze Current Image", 
                     command=self.analyze_current_image, height=35, width=200).pack(pady=5)
        ctk.CTkButton(analysis_tab, text="📁 Batch Process Directory", 
                     command=self.batch_process, height=35, width=200).pack(pady=5)
        ctk.CTkButton(analysis_tab, text="📁 Set Output Directory", 
                     command=self.set_output_directory, height=35, width=200).pack(pady=5)
        
        # Results tab
        results_tab = self.tabview.tab("📊 Results")
        
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
        """Analyze current image - DEMO"""
        if not self.current_images:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        current_image_data = self.current_images[self.current_image_index]
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
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        result_text = f"""
🔍 ANALYSIS COMPLETE - {timestamp}

📁 FILE: {filename}
🤖 METHOD: Demo Analysis
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

---
"""
        
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", result_text)
        
        self.status_label.configure(text=f"Analysis complete for {filename}")
        messagebox.showinfo("Analysis Complete", f"Demo analysis completed for {filename}")
    
    def batch_process(self):
        """Batch process - DEMO"""
        if not self.current_images:
            messagebox.showwarning("Warning", "Please load images first using 'Load Multiple Images'")
            return
        
        total = len(self.current_images)
        messagebox.showinfo("Batch Processing", f"Demo batch processing of {total} images completed!")
        self.status_label.configure(text=f"Batch processed {total} images")
    
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
            self.display_image_with_zoom(current_image_data['analyzed'], self.analyzed_image_label)
        else:
            self.analyzed_image_label.configure(image=None, text="Analysis result will appear here\n\nAfter running analysis")
    
    def display_image_with_zoom(self, pil_image, label_widget):
        """Display image with IMPROVED width fitting"""
        if pil_image is None:
            return
        
        try:
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
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    app = TestDamageAnalyzer()
    app.run()

if __name__ == "__main__":
    main()