# Logo and icon utilities for the Structural Damage Analyzer
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
import io
import base64

def create_app_logo(size=(64, 64)):
    """Create a modern logo for the application"""
    # Create a new image with transparent background
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple, clean logo
    margin = 4
    circle_bbox = [margin, margin, size[0]-margin, size[1]-margin]
    draw.ellipse(circle_bbox, fill=LOGO_COLORS['bg'], outline=LOGO_COLORS['accent'], width=2)
    
    # Inner design - representing structural analysis
    center_x, center_y = size[0]//2, size[1]//2
    
    # Draw grid pattern (representing structure)
    grid_color = LOGO_COLORS['success']
    for i in range(3):
        for j in range(3):
            x = margin + 8 + i * 12
            y = margin + 8 + j * 12
            draw.rectangle([x, y, x+8, y+8], outline=grid_color, width=1)
    
    # Draw crack pattern (representing damage detection)
    crack_color = LOGO_COLORS['error']
    # Zigzag crack line
    points = [
        (center_x - 15, center_y - 10),
        (center_x - 8, center_y - 5),
        (center_x - 2, center_y + 2),
        (center_x + 5, center_y - 3),
        (center_x + 12, center_y + 8)
    ]
    
    for i in range(len(points)-1):
        draw.line([points[i], points[i+1]], fill=crack_color, width=2)
    
    return img

def get_logo_base64():
    """Get logo as base64 string for embedding"""
    logo = create_app_logo()
    buffer = io.BytesIO()
    logo.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# Simple color scheme for the logo
LOGO_COLORS = {
    'bg': '#2b2b2b',
    'accent': '#0078d4',
    'success': '#107c10',
    'error': '#d13438',
}

# Button color schemes for different actions
BUTTON_COLORS = {
    'primary': {'bg': '#5E81AC', 'fg': '#ECEFF4', 'active_bg': '#81A1C1'},
    'success': {'bg': '#A3BE8C', 'fg': '#2E3440', 'active_bg': '#B5C9A4'},
    'warning': {'bg': '#EBCB8B', 'fg': '#2E3440', 'active_bg': '#F0D5A0'},
    'danger': {'bg': '#BF616A', 'fg': '#ECEFF4', 'active_bg': '#D08770'},
    'info': {'bg': '#88C0D0', 'fg': '#2E3440', 'active_bg': '#9FCAE0'},
    'secondary': {'bg': '#4C566A', 'fg': '#ECEFF4', 'active_bg': '#5E6B7D'},
}