import tkinter as tk
from tkinter import ttk, messagebox
import markdown
import os
from pathlib import Path
import webbrowser
from tkhtmlview import HTMLLabel
import pdfkit
from markdown.extensions import fenced_code, tables
import json
import datetime

class DocsViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ZMTech Finance Documentation")
        self.root.geometry("1400x900")
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Initialize variables
        self.current_section = "Home"
        self.history = []
        self.history_position = -1
        
        # Load documentation
        self.load_documentation()
        
        # Setup UI
        self.setup_ui()
        
        # Load initial page
        self.load_section("Home")
        
    def load_documentation(self):
        """Load all documentation content"""
        self.docs = {
            "Home": self.load_doc_file("home.md"),
            "User Manual": self.load_doc_file("user_manual.md"),
            "Technical Guide": self.load_doc_file("technical_guide.md"),
            "API Reference": self.load_doc_file("api_reference.md"),
            "Examples": self.load_doc_file("examples.md"),
            "Troubleshooting": self.load_doc_file("troubleshooting.md"),
            "Quick Start": self.load_doc_file("quick_start.md")
        }
        
    def load_doc_file(self, filename):
        """Load documentation from file or return default content"""
        try:
            with open(f"docs/{filename}", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return self.get_default_content(filename)
            
    def setup_ui(self):
        """Setup the main user interface"""
        # Main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create UI components
        self.create_toolbar()
        self.create_sidebar()
        self.create_content_area()
        self.create_status_bar()
        
    def create_toolbar(self):
        """Create the top toolbar"""
        toolbar = ttk.Frame(self.main_container)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Navigation buttons
        ttk.Button(toolbar, text="←", command=self.go_back).pack(side="left", padx=2)
        ttk.Button(toolbar, text="→", command=self.go_forward).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Home", command=lambda: self.load_section("Home")).pack(side="left", padx=5)
        
        # Search bar
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(toolbar, textvariable=self.search_var)
        search_entry.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(toolbar, text="Search", command=self.search_docs).pack(side="left", padx=5)
        
        # Export buttons
        ttk.Button(toolbar, text="PDF", command=self.export_pdf).pack(side="right", padx=2)
        ttk.Button(toolbar, text="HTML", command=self.export_html).pack(side="right", padx=2)
        
    def create_sidebar(self):
        """Create the navigation sidebar"""
        sidebar = ttk.Frame(self.main_container, relief="solid", borderwidth=1)
        sidebar.grid(row=1, column=0, sticky="ns", padx=(0, 10))
        
        # Section buttons
        sections = list(self.docs.keys())
        for i, section in enumerate(sections):
            ttk.Button(
                sidebar,
                text=section,
                command=lambda s=section: self.load_section(s)
            ).grid(row=i, column=0, pady=2, padx=5, sticky="ew")
            
    def create_content_area(self):
        """Create the main content area"""
        content_frame = ttk.Frame(self.main_container)
        content_frame.grid(row=1, column=1, sticky="nsew")
        
        # HTML viewer
        self.content_viewer = HTMLLabel(content_frame)
        self.content_viewer.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure scrolling
        self.content_viewer.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(command=self.content_viewer.yview)
        
        # Configure grid weights
        self.main_container.grid_columnconfigure(1, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        
    def create_status_bar(self):
        """Create the bottom status bar"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.main_container, textvariable=self.status_var)
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
    def load_section(self, section):
        """Load a documentation section"""
        try:
            self.current_section = section
            content = self.docs.get(section, "# Error\nSection not found")
            
            # Convert markdown to HTML
            html_content = markdown.markdown(
                content,
                extensions=['fenced_code', 'tables']
            )
            
            # Update viewer
            self.content_viewer.set_html(html_content)
            
            # Update history
            self.add_to_history(section)
            
            # Update status
            self.status_var.set(f"Viewing: {section}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load section: {str(e)}")
            
    def search_docs(self):
        """Search documentation content"""
        query = self.search_var.get().lower()
        if not query:
            return
            
        results = []
        for section, content in self.docs.items():
            if query in content.lower():
                excerpt = self.get_search_excerpt(content, query)
                results.append(f"## {section}\n{excerpt}\n")
                
        if results:
            search_content = "# Search Results\n\n" + "\n".join(results)
            html_content = markdown.markdown(search_content)
            self.content_viewer.set_html(html_content)
            self.status_var.set(f"Search results for: {query}")
        else:
            messagebox.showinfo("Search", "No results found")
            
    def get_search_excerpt(self, content, query, context=50):
        """Get excerpt of text around search query"""
        idx = content.lower().find(query)
        if idx == -1:
            return ""
            
        start = max(0, idx - context)
        end = min(len(content), idx + len(query) + context)
        
        excerpt = content[start:end]
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."
            
        return excerpt
        
    def add_to_history(self, section):
        """Add section to navigation history"""
        self.history_position += 1
        self.history = self.history[:self.history_position]
        self.history.append(section)
        
    def go_back(self):
        """Navigate backwards in history"""
        if self.history_position > 0:
            self.history_position -= 1
            self.load_section(self.history[self.history_position])
            
    def go_forward(self):
        """Navigate forwards in history"""
        if self.history_position < len(self.history) - 1:
            self.history_position += 1
            self.load_section(self.history[self.history_position])
            
    def export_pdf(self):
        """Export current section to PDF"""
        try:
            filename = f"docs/pdf/{self.current_section}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            # Convert markdown to HTML
            html_content = markdown.markdown(
                self.docs[self.current_section],
                extensions=['fenced_code', 'tables']
            )
            
            # Convert HTML to PDF
            pdfkit.from_string(html_content, filename)
            
            messagebox.showinfo("Export", f"PDF exported to: {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export PDF: {str(e)}")
            
    def export_html(self):
        """Export current section to HTML"""
        try:
            filename = f"docs/html/{self.current_section}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            # Convert markdown to HTML
            html_content = markdown.markdown(
                self.docs[self.current_section],
                extensions=['fenced_code', 'tables']
            )
            
            # Save HTML file
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            # Open in browser
            webbrowser.open(filename)
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export HTML: {str(e)}")
            
    def get_default_content(self, filename):
        """Get default content for documentation files"""
        defaults = {
            "home.md": """# Welcome to ZMTech Finance
            
## Quick Links
- [User Manual](#user-manual)
- [Technical Guide](#technical-guide)
- [API Reference](#api-reference)
- [Examples](#examples)
            """,
            
            "user_manual.md": """# User Manual
            
## Installation
1. Install Python 3.8+
2. Install required packages
3. Run the application
            """,
            # Add more default content for other files
        }
        return defaults.get(filename, "# Documentation\nContent coming soon...")
        
    def run(self):
        """Run the documentation viewer"""
        self.root.mainloop()

def main():
    viewer = DocsViewer()
    viewer.run()

if __name__ == "__main__":
    main()