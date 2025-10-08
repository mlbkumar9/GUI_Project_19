#!/usr/bin/env python3
"""
Installation script for the modern UI version
"""

import subprocess
import sys

def install_customtkinter():
    """Install CustomTkinter for modern UI"""
    try:
        print("🚀 Installing CustomTkinter for modern UI...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter>=5.2.0"])
        print("✅ CustomTkinter installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install CustomTkinter: {e}")
        return False

def test_installation():
    """Test if CustomTkinter is working"""
    try:
        import customtkinter as ctk
        print("✅ CustomTkinter is working correctly!")
        
        # Show version
        print(f"📦 CustomTkinter version: {ctk.__version__}")
        return True
    except ImportError as e:
        print(f"❌ CustomTkinter import failed: {e}")
        return False

def main():
    """Main installation process"""
    print("🎨 Modern UI Setup for Structural Damage Analyzer")
    print("=" * 50)
    
    # Install CustomTkinter
    if install_customtkinter():
        print()
        if test_installation():
            print()
            print("🎉 Installation complete!")
            print("📋 Next steps:")
            print("   1. Run: python app_modern.py")
            print("   2. Enjoy the modern, professional UI!")
            print()
            print("✨ Features of the modern UI:")
            print("   • Native dark theme that actually works")
            print("   • Professional, modern appearance")
            print("   • Proper font sizes and readability")
            print("   • Consistent styling throughout")
            print("   • Better user experience")
        else:
            print("❌ Installation verification failed")
            return False
    else:
        print("❌ Installation failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)