import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
from datetime import datetime
import sys
from pathlib import Path

class StockAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Earnings Analysis")
        
        # Set window size and position
        window_width = 600
        window_height = 400
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Stock ticker inputs
        ttk.Label(main_frame, text="Enter Stock Tickers:").grid(row=0, column=0, columnspan=2, pady=10)
        
        # Ticker 1
        ttk.Label(main_frame, text="Ticker 1:").grid(row=1, column=0, padx=5, pady=5)
        self.ticker1 = ttk.Entry(main_frame, width=10)
        self.ticker1.grid(row=1, column=1, padx=5, pady=5)
        
        # Ticker 2
        ttk.Label(main_frame, text="Ticker 2:").grid(row=2, column=0, padx=5, pady=5)
        self.ticker2 = ttk.Entry(main_frame, width=10)
        self.ticker2.grid(row=2, column=1, padx=5, pady=5)
        
        # Analysis period
        ttk.Label(main_frame, text="Analysis Settings:").grid(row=3, column=0, columnspan=2, pady=10)
        
        # Days before/after earnings
        ttk.Label(main_frame, text="Days Before:").grid(row=4, column=0, padx=5, pady=5)
        self.days_before = ttk.Entry(main_frame, width=5)
        self.days_before.insert(0, "5")
        self.days_before.grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Days After:").grid(row=5, column=0, padx=5, pady=5)
        self.days_after = ttk.Entry(main_frame, width=5)
        self.days_after.insert(0, "5")
        self.days_after.grid(row=5, column=1, padx=5, pady=5)
        
        # Analyze button
        self.analyze_button = ttk.Button(main_frame, text="Analyze", command=self.run_analysis)
        self.analyze_button.grid(row=6, column=0, columnspan=2, pady=20)
        
        # Progress and status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.grid(row=7, column=0, columnspan=2, pady=10)
        
        # Output display
        self.output_text = tk.Text(main_frame, height=10, width=50)
        self.output_text.grid(row=8, column=0, columnspan=2, pady=10)
        
        # Add scrollbar to output
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.output_text.yview)
        scrollbar.grid(row=8, column=2, sticky="ns")
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
    def validate_inputs(self):
        """Validate user inputs"""
        if not self.ticker1.get() or not self.ticker2.get():
            messagebox.showerror("Error", "Please enter both stock tickers")
            return False
        
        try:
            days_before = int(self.days_before.get())
            days_after = int(self.days_after.get())
            if days_before < 1 or days_after < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Days before/after must be positive integers")
            return False
        
        return True
    
    def run_analysis(self):
        """Run the analysis script"""
        if not self.validate_inputs():
            return
        
        self.analyze_button.state(['disabled'])
        self.status_var.set("Analyzing...")
        self.output_text.delete(1.0, tk.END)
        self.root.update()
        
        try:
            # Get the path to az.py
            script_path = Path(__file__).parent / "az.py"
            
            # Run the analysis script
            process = subprocess.Popen(
                [sys.executable, str(script_path), 
                 self.ticker1.get().upper(), 
                 self.ticker2.get().upper()],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Get output
            output, error = process.communicate()
            
            # Display output
            if output:
                self.output_text.insert(tk.END, output)
            if error:
                self.output_text.insert(tk.END, "\nErrors:\n" + error)
            
            self.status_var.set("Analysis complete")
            
            # Open output directory if analysis was successful
            output_dir = f"earnings_analysis_{self.ticker1.get().upper()}_{self.ticker2.get().upper()}_{datetime.now().strftime('%Y%m%d')}"
            if os.path.exists(output_dir):
                os.startfile(output_dir)
            
        except Exception as e:
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", str(e))
        
        finally:
            self.analyze_button.state(['!disabled'])

def main():
    root = tk.Tk()
    app = StockAnalyzerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()