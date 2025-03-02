from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import csv

# Import your existing analysis code
from az import analyze_earnings_impact

class ZMTechApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ZMTech Finance - Stock Analysis")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI"""
        # Configure main window
        window_width = 1000
        window_height = 800
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Create main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create frames
        self.create_input_frame()
        self.create_output_frame()
        
    def create_input_frame(self):
        """Create input section"""
        input_frame = ttk.LabelFrame(self.main_container, text="Analysis Input", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Title
        title_label = ttk.Label(input_frame, text="Stock Analysis", 
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Stock inputs
        for i in range(6):
            ttk.Label(input_frame, text=f"Ticker {i+1}:").grid(row=i+1, column=0, padx=5, pady=5)
            ticker_entry = ttk.Entry(input_frame, width=15)
            ticker_entry.grid(row=i+1, column=1, padx=5, pady=5)
            setattr(self, f'ticker{i+1}', ticker_entry)

        # Analysis period
        period_frame = ttk.LabelFrame(input_frame, text="Analysis Period", padding="5")
        period_frame.grid(row=7, column=0, columnspan=4, sticky="ew", pady=10)
        
        ttk.Label(period_frame, text="Days Before:").grid(row=0, column=0, padx=5)
        self.days_before = ttk.Entry(period_frame, width=10)
        self.days_before.insert(0, "10")
        self.days_before.grid(row=0, column=1, padx=5)
        
        ttk.Label(period_frame, text="Days After:").grid(row=0, column=2, padx=5)
        self.days_after = ttk.Entry(period_frame, width=10)
        self.days_after.insert(0, "10")
        self.days_after.grid(row=0, column=3, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=8, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="Run Analysis", 
                  command=self.run_analysis).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear", 
                  command=self.clear_output).grid(row=0, column=1, padx=5)
        
    def create_output_frame(self):
        """Create output section"""
        output_frame = ttk.LabelFrame(self.main_container, text="Analysis Output", padding="10")
        output_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(output_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        # Output text
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=20)
        self.output_text.grid(row=1, column=0, sticky="nsew")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", 
                                command=self.output_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure output frame grid
        output_frame.grid_rowconfigure(1, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)
        
    def create_table_row(self, labels, values, indent=0):
        """Create a formatted table row"""
        indent_str = " " * indent
        row = ""
        for label, value in zip(labels, values):
            row += f"{indent_str}{label:<20}: {str(value):<15}  "
        return row + "\n"

    def export_to_csv(self, results_data):
        """Export analysis results to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"earnings_analysis_{timestamp}.csv"
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow([
                    'Ticker Pair',
                    'RSI Level (Stock1)',
                    'RSI Level (Stock2)',
                    'Close (Stock1)',
                    'Close (Stock2)',
                    'Volume (Stock1)',
                    'Volume (Stock2)',
                    'Returns (Stock1)',
                    'Returns (Stock2)'
                ])
                
                # Debug print
                self.output_text.insert(tk.END, "\nDebug - CSV Export Data:\n")
                for row in results_data:
                    self.output_text.insert(tk.END, f"Row data: {row}\n")
                    writer.writerow(row)
            
            self.output_text.insert(tk.END, f"\nResults exported to: {filename}\n")
            return filename
            
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self.status_var.set(error_msg)
            self.output_text.insert(tk.END, f"\n{error_msg}\n")
            import traceback
            self.output_text.insert(tk.END, f"Traceback:\n{traceback.format_exc()}\n")
            return None

    def run_analysis(self):
        """Run the stock analysis"""
        try:
            self.status_var.set("Running analysis...")
            self.root.update()
            
            # Get input values
            tickers = []
            for i in range(6):
                ticker = getattr(self, f'ticker{i+1}').get().upper()
                if ticker:  # Only add non-empty tickers
                    tickers.append(ticker)
            
            if len(tickers) < 2:
                raise ValueError("Please enter at least two stock tickers")
            
            days_before = int(self.days_before.get())
            days_after = int(self.days_after.get())
            
            # Clear previous results
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Analysis for {len(tickers)} stocks: {', '.join(tickers)}\n")
            self.output_text.insert(tk.END, f"Period: {days_before} days before to {days_after} days after earnings\n\n")
            
            # Initialize results_data list
            results_data = []
            
            def format_value(value):
                """Format value for display"""
                try:
                    if isinstance(value, pd.Series):
                        non_null = value.dropna()
                        if not non_null.empty:
                            last_value = non_null.iloc[-1]
                            return str(last_value)
                    elif isinstance(value, (int, float)):
                        return f"{value:.2f}"
                    elif isinstance(value, str):
                        return value
                    elif value is not None:
                        return str(value)
                except Exception as e:
                    self.output_text.insert(tk.END, f"Debug - Format error: {str(e)}\n")
                return "No data"
            
            # Analyze each pair of tickers
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    pair_name = f"{tickers[i]}-{tickers[j]}"
                    self.output_text.insert(tk.END, f"\n{'='*80}\n")
                    self.output_text.insert(tk.END, f"Analysis Results for {tickers[i]} and {tickers[j]}\n")
                    self.output_text.insert(tk.END, f"{'='*80}\n\n")
                    
                    results = analyze_earnings_impact(
                        ticker1=tickers[i],
                        ticker2=tickers[j],
                        days_before=days_before,
                        days_after=days_after
                    )
                    
                    # Create headers
                    header_row = f"{'Metric':<30} {tickers[i]:<25} {tickers[j]:<25}\n"
                    self.output_text.insert(tk.END, header_row)
                    self.output_text.insert(tk.END, "-" * 80 + "\n")
                    
                    # Define metrics with their corresponding keys
                    metrics = {
                        "RSI Level": ("RSI_Level", "RSI_Level"),
                        "Close": ("Close", "Close"),
                        "Volume": ("Volume", "Volume"),
                        "Returns": ("Returns", "Returns")
                    }
                    
                    # Process and store row data
                    row_data = [pair_name]  # Start with the ticker pair
                    for _, (key1, key2) in metrics.items():
                        val1 = format_value(results.get(key1))
                        val2 = format_value(results.get(key2))
                        row_data.extend([val1, val2])
                        
                    # Add data for CSV export
                    results_data.append(row_data)
                    
                    # Display in UI
                    for metric_name, (key1, key2) in metrics.items():
                        val1 = row_data[len(row_data)-8]  # Get values from row_data
                        val2 = row_data[len(row_data)-7]
                        row = f"{metric_name:<30} {val1:<25} {val2:<25}\n"
                        self.output_text.insert(tk.END, row)
                    
                    self.output_text.see(tk.END)
                    self.root.update()
            
            # Export results to CSV
            if results_data:
                csv_file = self.export_to_csv(results_data)
                if csv_file:
                    self.output_text.insert(tk.END, f"\nResults exported to: {csv_file}\n")
            else:
                self.output_text.insert(tk.END, "\nNo data to export\n")
            
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.output_text.insert(tk.END, f"\nError occurred: {str(e)}\n")
            import traceback
            self.output_text.insert(tk.END, f"\nTraceback:\n{traceback.format_exc()}\n")
        
    def clear_output(self):
        """Clear the output text"""
        self.output_text.delete(1.0, tk.END)
        self.status_var.set("Ready")
            
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    app = ZMTechApp()
    app.run()

if __name__ == "__main__":
    main()