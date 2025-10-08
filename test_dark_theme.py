#!/usr/bin/env python3
"""
Test script to verify the simplified dark theme
"""

import tkinter as tk
from tkinter import ttk
from utils.ui_theme import SimpleTheme, SimpleButton, SimpleText, DARK_COLORS

def test_simple_theme():
    """Test the simplified dark theme"""
    root = tk.Tk()
    root.title("Simple Dark Theme Test")
    root.geometry("700x500")
    
    # Apply simple theme
    theme = SimpleTheme(root)
    
    # Test frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Test large, readable label
    ttk.Label(main_frame, text="Simple Dark Theme Test", 
             font=('Segoe UI', 18, 'bold')).pack(pady=10)
    
    # Test combobox with larger font
    ttk.Label(main_frame, text="Framework Selection:").pack(anchor='w', pady=(10, 5))
    framework_var = tk.StringVar(value="PyTorch")
    combo = ttk.Combobox(main_frame, textvariable=framework_var, 
                        values=["PyTorch", "Keras", "TensorFlow", "OpenCV"], 
                        state="readonly", width=25)
    combo.pack(pady=5, anchor='w')
    
    # Test entry with larger font
    ttk.Label(main_frame, text="Text Entry:").pack(anchor='w', pady=(10, 5))
    entry = ttk.Entry(main_frame, width=40)
    entry.pack(pady=5, anchor='w')
    entry.insert(0, "Test readable text entry")
    
    # Test buttons
    button_frame = tk.Frame(main_frame, bg=DARK_COLORS['bg'])
    button_frame.pack(fill=tk.X, pady=10)
    
    SimpleButton(button_frame, text="Primary", style='primary').pack(side=tk.LEFT, padx=(0, 10))
    SimpleButton(button_frame, text="Success", style='success').pack(side=tk.LEFT, padx=(0, 10))
    SimpleButton(button_frame, text="Warning", style='warning').pack(side=tk.LEFT, padx=(0, 10))
    SimpleButton(button_frame, text="Danger", style='danger').pack(side=tk.LEFT)
    
    # Test text widget with larger font
    ttk.Label(main_frame, text="Text Area:").pack(anchor='w', pady=(10, 5))
    text_frame = tk.Frame(main_frame, bg=DARK_COLORS['bg'])
    text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    text_widget = SimpleText(text_frame, height=10)
    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    text_widget.insert(tk.END, """Simple Dark Theme Test

✅ Features:
• Larger, more readable fonts (11pt)
• High contrast colors
• Simple, clean design
• Consistent dark theme
• Better button styling

This text should be clearly readable with good contrast.
The font size is now larger and more comfortable to read.
""")
    
    print("🎨 Simple Dark Theme Test Launched")
    print("✅ Improvements:")
    print("   • Larger fonts (11pt) for better readability")
    print("   • High contrast colors")
    print("   • Simplified theme system")
    print("   • Consistent dark background")
    print("   • Better button styling")
    
    root.mainloop()

if __name__ == "__main__":
    test_simple_theme()