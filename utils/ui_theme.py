import tkinter as tk
from tkinter import ttk

# Simple, effective dark theme colors
DARK_COLORS = {
    'bg': '#2b2b2b',           # Main background
    'fg': '#ffffff',           # Main text
    'entry_bg': '#3c3c3c',     # Entry/input background
    'entry_fg': '#ffffff',     # Entry text
    'button_bg': '#404040',    # Button background
    'accent': '#0078d4',       # Accent color (blue)
    'success': '#107c10',      # Success green
    'warning': '#ff8c00',      # Warning orange
    'error': '#d13438',        # Error red
    'info': '#00bcf2',         # Info cyan
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

class SimpleTheme:
    """Simple, effective dark theme"""
    
    def __init__(self, root):
        self.root = root
        self.apply_theme()
    
    def apply_theme(self):
        """Apply simple dark theme"""
        # Set root background
        self.root.configure(bg=DARK_COLORS['bg'])
        
        # Create style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure basic styles with larger fonts
        self.style.configure('TLabel',
                           background=DARK_COLORS['bg'],
                           foreground=DARK_COLORS['fg'],
                           font=('Segoe UI', 11))
        
        self.style.configure('TFrame',
                           background=DARK_COLORS['bg'])
        
        self.style.configure('TLabelFrame',
                           background=DARK_COLORS['bg'],
                           foreground=DARK_COLORS['fg'],
                           font=('Segoe UI', 11, 'bold'))
        
        self.style.configure('TLabelFrame.Label',
                           background=DARK_COLORS['bg'],
                           foreground=DARK_COLORS['accent'],
                           font=('Segoe UI', 11, 'bold'))
        
        # Entry widgets
        self.style.configure('TEntry',
                           fieldbackground=DARK_COLORS['entry_bg'],
                           foreground=DARK_COLORS['entry_fg'],
                           insertcolor=DARK_COLORS['fg'],
                           font=('Segoe UI', 11))
        
        # Combobox - simple and effective
        self.style.configure('TCombobox',
                           fieldbackground=DARK_COLORS['entry_bg'],
                           foreground=DARK_COLORS['entry_fg'],
                           arrowcolor=DARK_COLORS['fg'],
                           font=('Segoe UI', 11))
        
        # Configure combobox dropdown
        self.root.option_add('*TCombobox*Listbox.Background', DARK_COLORS['entry_bg'])
        self.root.option_add('*TCombobox*Listbox.Foreground', DARK_COLORS['entry_fg'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', DARK_COLORS['accent'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', DARK_COLORS['fg'])
        self.root.option_add('*TCombobox*Listbox.font', 'Segoe UI 11')
        
        # Notebook tabs
        self.style.configure('TNotebook',
                           background=DARK_COLORS['bg'])
        
        self.style.configure('TNotebook.Tab',
                           background=DARK_COLORS['button_bg'],
                           foreground=DARK_COLORS['fg'],
                           padding=[15, 8],
                           font=('Segoe UI', 11, 'bold'))
        
        self.style.map('TNotebook.Tab',
                      background=[('selected', DARK_COLORS['accent'])])
        
        # Progressbar
        self.style.configure('TProgressbar',
                           background=DARK_COLORS['accent'],
                           troughcolor=DARK_COLORS['button_bg'])
        
        # Scrollbar
        self.style.configure('TScrollbar',
                           background=DARK_COLORS['button_bg'],
                           troughcolor=DARK_COLORS['bg'],
                           arrowcolor=DARK_COLORS['fg'])

class ModernButton(tk.Button):
    """Custom modern button with hover effects"""
    
    def __init__(self, parent, text="", command=None, style='primary', **kwargs):
        self.style_name = style
        self.colors = BUTTON_COLORS.get(style, BUTTON_COLORS['primary'])
        
        # Default button configuration
        default_config = {
            'text': text,
            'command': command,
            'bg': self.colors['bg'],
            'fg': self.colors['fg'],
            'activebackground': self.colors['active_bg'],
            'activeforeground': self.colors['fg'],
            'relief': 'flat',
            'borderwidth': 0,
            'font': ('Segoe UI', 10, 'bold'),
            'cursor': 'hand2',
            'padx': 20,
            'pady': 8
        }
        
        # Update with any custom kwargs
        default_config.update(kwargs)
        
        super().__init__(parent, **default_config)
        
        # Bind hover effects
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        
        # Store original colors
        self.original_bg = self.colors['bg']
        self.hover_bg = self.colors['active_bg']
    
    def on_enter(self, event):
        """Handle mouse enter (hover)"""
        self.configure(bg=self.hover_bg)
    
    def on_leave(self, event):
        """Handle mouse leave"""
        self.configure(bg=self.original_bg)

class ModernText(tk.Text):
    """Custom text widget with dark theme"""
    
    def __init__(self, parent, **kwargs):
        default_config = {
            'bg': DARK_COLORS['entry_bg'],
            'fg': DARK_COLORS['entry_fg'],
            'insertbackground': DARK_COLORS['fg'],
            'selectbackground': DARK_COLORS['accent'],
            'selectforeground': DARK_COLORS['fg'],
            'relief': 'flat',
            'borderwidth': 1,
            'font': ('Consolas', 10),
            'wrap': tk.WORD
        }
        
        default_config.update(kwargs)
        super().__init__(parent, **default_config)

class SimpleButton(tk.Button):
    """Simple button with dark theme"""
    
    def __init__(self, parent, text="", command=None, style='primary', **kwargs):
        colors = {
            'primary': {'bg': DARK_COLORS['accent'], 'fg': 'white'},
            'success': {'bg': DARK_COLORS['success'], 'fg': 'white'},
            'warning': {'bg': DARK_COLORS['warning'], 'fg': 'white'},
            'danger': {'bg': DARK_COLORS['error'], 'fg': 'white'},
            'info': {'bg': DARK_COLORS['info'], 'fg': 'white'},
            'secondary': {'bg': DARK_COLORS['button_bg'], 'fg': 'white'},
        }
        
        color = colors.get(style, colors['primary'])
        
        super().__init__(parent, 
                        text=text,
                        command=command,
                        bg=color['bg'],
                        fg=color['fg'],
                        font=('Segoe UI', 11, 'bold'),
                        relief='flat',
                        borderwidth=0,
                        padx=20,
                        pady=8,
                        cursor='hand2',
                        **kwargs)

class SimpleText(tk.Text):
    """Simple text widget with dark theme"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent,
                        bg=DARK_COLORS['entry_bg'],
                        fg=DARK_COLORS['entry_fg'],
                        insertbackground=DARK_COLORS['fg'],
                        selectbackground=DARK_COLORS['accent'],
                        selectforeground='white',
                        font=('Consolas', 11),
                        relief='flat',
                        borderwidth=1,
                        **kwargs)

def create_status_bar(parent, textvariable):
    """Create a simple status bar"""
    status_frame = tk.Frame(parent, bg=DARK_COLORS['button_bg'], height=30)
    status_label = tk.Label(status_frame, 
                           textvariable=textvariable,
                           bg=DARK_COLORS['button_bg'],
                           fg=DARK_COLORS['fg'],
                           font=('Segoe UI', 10),
                           anchor='w',
                           padx=10)
    status_label.pack(fill=tk.BOTH, expand=True)
    return status_frame