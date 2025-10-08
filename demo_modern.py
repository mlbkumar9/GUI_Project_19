#!/usr/bin/env python3
"""
Demo script for the modern UI version using CustomTkinter
"""

def main():
    """Launch the modern UI demo"""
    print("🎨 Launching Modern Structural Damage Analyzer")
    print("=" * 50)
    print("✨ Modern UI Features:")
    print("   • Native dark theme that works perfectly")
    print("   • Professional, clean design")
    print("   • Proper font sizes and readability")
    print("   • Modern buttons and controls")
    print("   • Consistent styling throughout")
    print("   • Better user experience")
    print()
    print("🚀 Starting application...")
    
    try:
        from app_modern import ModernDamageAnalyzer
        app = ModernDamageAnalyzer()
        app.run()
    except ImportError as e:
        print("❌ Error: CustomTkinter not installed")
        print("📋 Please run: python install_modern_ui.py")
        print(f"   Error details: {e}")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()