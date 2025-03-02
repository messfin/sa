

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json
import sys
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import datetime
from docx import Document
import yfinance as yf
import pandas as pd

class ZMTechFinance:
    def __init__(self):
        self.setup_environment()
        self.load_configuration()
        self.initialize_ui()
        
    def setup_environment(self):
        """Setup directory structure and assets"""
        # Create necessary directories
        self.dirs = {
            'root': Path('zmtech_finance'),
            'assets': Path('zmtech_finance/assets'),
            'reports': Path('zmtech_finance/reports'),
            'data': Path('zmtech_finance/data'),
            'templates': Path('zmtech_finance/templates')
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def load_configuration(self):
        """Load application configuration"""
        self.config = {
            'colors': {
                'primary': '#1E3D59',
                'secondary': '#17B890',
                'accent': '#FFB400',
                'background': '#E8E8E8',
                'text': '#333333'
            },
            'fonts': {
                'heading': ('Montserrat', 12, 'bold'),
                'body': ('Open Sans', 10),
                'small': ('Open Sans', 8)
            }
        }
        
    def initialize_ui(self):
        """Initialize the main application UI"""
        self.root = tk.Tk()
        self.root.title("ZMTech Finance - Stock Analysis Suite")
        self.setup_ui_styles()
        self.create_main_window()
        
    def setup_ui_styles(self):
        """Configure UI styles and themes"""
        style = ttk.Style()
        
        # Configure colors
        style.configure('ZMTech.TFrame',
                       background=self.config['colors']['background'])
        style.configure('ZMTech.TLabel',
                       background=self.config['colors']['background'],
                       foreground=self.config['colors']['text'])
        style.configure('ZMTech.TButton',
                       background=self.config['colors']['primary'],
                       foreground='white')
                       
    def create_main_window(self):
        """Create the main application window"""
        # Main container
        self.main_frame = ttk.Frame(self.root, style='ZMTech.TFrame')
        self.main_frame.pack(fill='both', expand=True)
        
        # Header with logo
        self.create_header()
        
        # Navigation
        self.create_navigation()
        
        # Content area
        self.content_frame = ttk.Frame(self.main_frame, style='ZMTech.TFrame')
        self.content_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Status bar
        self.create_status_bar()
        
    def create_header(self):
        """Create application header"""
        header_frame = ttk.Frame(self.main_frame, style='ZMTech.TFrame')
        header_frame.pack(fill='x', padx=10, pady=5)
        
        # Logo
        logo_path = self.dirs['assets'] / 'zmtech_logo.png'
        if logo_path.exists():
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((200, 50))
            logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(header_frame, image=logo_photo)
            logo_label.image = logo_photo
            logo_label.pack(side='left')
            
    def create_navigation(self):
        """Create navigation menu"""
        nav_frame = ttk.Frame(self.main_frame, style='ZMTech.TFrame')
        nav_frame.pack(fill='x', padx=10, pady=5)
        
        # Navigation buttons
        ttk.Button(nav_frame, 
                  text="Stock Analysis",
                  command=self.show_stock_analysis).pack(side='left', padx=5)
        ttk.Button(nav_frame,
                  text="Reports",
                  command=self.show_reports).pack(side='left', padx=5)
        ttk.Button(nav_frame,
                  text="Settings",
                  command=self.show_settings).pack(side='left', padx=5)
                  
    def create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root,
                             textvariable=self.status_var,
                             relief='sunken',
                             style='ZMTech.TLabel')
        status_bar.pack(side='bottom', fill='x')
        
    def show_stock_analysis(self):
        """Show stock analysis interface"""
        self.clear_content()
        
        # Create analysis form
        form_frame = ttk.Frame(self.content_frame, style='ZMTech.TFrame')
        form_frame.pack(padx=20, pady=20)
        
        # Stock inputs
        ttk.Label(form_frame,
                 text="Enter Stock Tickers:",
                 style='ZMTech.TLabel').grid(row=0, column=0, columnspan=2)
        
        # Ticker 1
        ttk.Label(form_frame,
                 text="Ticker 1:",
                 style='ZMTech.TLabel').grid(row=1, column=0)
        self.ticker1_var = tk.StringVar()
        ttk.Entry(form_frame,
                 textvariable=self.ticker1_var).grid(row=1, column=1)
        
        # Ticker 2
        ttk.Label(form_frame,
                 text="Ticker 2:",
                 style='ZMTech.TLabel').grid(row=2, column=0)
        self.ticker2_var = tk.StringVar()
        ttk.Entry(form_frame,
                 textvariable=self.ticker2_var).grid(row=2, column=1)
        
        # Analysis button
        ttk.Button(form_frame,
                  text="Analyze",
                  command=self.run_analysis).grid(row=3, column=0, columnspan=2)
                  
    def run_analysis(self):
        """Run stock analysis"""
        # ... Analysis code ...
        pass
        
    def show_reports(self):
        """Show reports interface"""
        self.clear_content()
        # ... Reports interface code ...
        
    def show_settings(self):
        """Show settings interface"""
        self.clear_content()
        # ... Settings interface code ...
        
    def clear_content(self):
        """Clear content frame"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    app = ZMTechFinance()
    app.run()

if __name__ == "__main__":
    main()
```