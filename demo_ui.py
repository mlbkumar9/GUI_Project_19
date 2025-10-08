#!/usr/bin/env python3
"""
Demo script to showcase the modern UI of the Structural Damage Analyzer
Run this to see the clean, readable, and modern interface
"""

import tkinter as tk
from app import StructuralDamageAnalyzer

def main():
    """Launch the modern UI demo"""
    print("🚀 Launching Structural Damage Analyzer - Clean & Modern Version")
    print("✨ Improvements:")
    print("   • Clean, readable code structure")
    print("   • Modern dark theme with professional styling")
    print("   • Colorful, intuitive button system")
    print("   • Organized modular design")
    print("   • Enhanced user experience")
    print("   • Better error handling and feedback")
    print("\n🎯 Ready to train models and analyze structural damage!")
    
    root = tk.Tk()
    app = StructuralDamageAnalyzer(root)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()