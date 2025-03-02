import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf
from StockAnalyzer import analyze_stock  # Import the previous StockAnalyzer code

class StockAnalyzerUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stock Technical Analysis Tool")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI components"""
        # Configure main window
        window_width = 1200
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
        
        # Stock ticker input
        ttk.Label(input_frame, text="Stock Ticker:").grid(row=0, column=0, padx=5, pady=5)
        self.ticker_entry = ttk.Entry(input_frame, width=10)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Date range inputs
        ttk.Label(input_frame, text="Start Date:").grid(row=1, column=0, padx=5, pady=5)
        self.start_date = ttk.Entry(input_frame, width=10)
        self.start_date.insert(0, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        self.start_date.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="End Date:").grid(row=1, column=2, padx=5, pady=5)
        self.end_date = ttk.Entry(input_frame, width=10)
        self.end_date.insert(0, datetime.now().strftime('%Y-%m-%d'))
        self.end_date.grid(row=1, column=3, padx=5, pady=5)
        
        # Technical indicators frame
        indicators_frame = ttk.LabelFrame(input_frame, text="Technical Indicators", padding="5")
        indicators_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=10)
        
        # Checkboxes for indicators
        self.show_ema = tk.BooleanVar(value=True)
        ttk.Checkbutton(indicators_frame, text="EMA", variable=self.show_ema).grid(row=0, column=0, padx=5)
        
        self.show_macd = tk.BooleanVar(value=True)
        ttk.Checkbutton(indicators_frame, text="MACD", variable=self.show_macd).grid(row=0, column=1, padx=5)
        
        self.show_rsi = tk.BooleanVar(value=True)
        ttk.Checkbutton(indicators_frame, text="RSI", variable=self.show_rsi).grid(row=0, column=2, padx=5)
        
        self.show_volume = tk.BooleanVar(value=True)
        ttk.Checkbutton(indicators_frame, text="Volume", variable=self.show_volume).grid(row=0, column=3, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="Analyze", command=self.run_analysis).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_output).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save Plot", command=self.save_plot).grid(row=0, column=2, padx=5)
        
    def create_output_frame(self):
        """Create output section"""
        output_frame = ttk.LabelFrame(self.main_container, text="Analysis Output", padding="10")
        output_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(output_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Chart")
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Summary text widget
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD, height=20)
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        
        # Configure output frame grid
        output_frame.grid_rowconfigure(0, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)
        
    def run_analysis(self):
        """Run the stock analysis"""
        try:
            ticker = self.ticker_entry.get().upper()
            if not ticker:
                messagebox.showerror("Error", "Please enter a stock ticker")
                return
            
            # Clear previous outputs
            for widget in self.chart_frame.winfo_children():
                widget.destroy()
            self.summary_text.delete(1.0, tk.END)
            
            # Run analysis
            summary, fig = analyze_stock(
                ticker,
                start_date=self.start_date.get(),
                end_date=self.end_date.get()
            )
            
            # Display chart
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            
            # Display summary
            self.summary_text.insert(tk.END, "Technical Analysis Summary:\n\n")
            for key, value in summary.items():
                self.summary_text.insert(tk.END, f"{key}: {value}\n")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def clear_output(self):
        """Clear all outputs"""
        self.ticker_entry.delete(0, tk.END)
        self.start_date.delete(0, tk.END)
        self.start_date.insert(0, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        self.end_date.delete(0, tk.END)
        self.end_date.insert(0, datetime.now().strftime('%Y-%m-%d'))
        
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        self.summary_text.delete(1.0, tk.END)
    
    def save_plot(self):
        """Save the current plot"""
        try:
            ticker = self.ticker_entry.get().upper()
            if not ticker:
                messagebox.showerror("Error", "Please run analysis first")
                return
            
            filename = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename)
            messagebox.showinfo("Success", f"Plot saved as {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    app = StockAnalyzerUI()
    app.run()

if __name__ == "__main__":
    main()