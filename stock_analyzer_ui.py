import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf
import os
from pathlib import Path
from StockAnalyzer import analyze_stock

class StockAnalyzerUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stock Technical Analysis Tool")
        
        # Configure default paths
        self.config = {
            'save_dir': Path('saved_plots'),
            'default_dpi': 300,
        }
        
        # Create default directories
        self.config['save_dir'].mkdir(exist_ok=True)
        
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
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="Analyze", command=self.run_analysis).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_output).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save Plot", command=self.save_plot).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Set Save Location", command=self.set_save_location).grid(row=0, column=3, padx=5)

    def create_output_frame(self):
        """Create output section"""
        output_frame = ttk.LabelFrame(self.main_container, text="Analysis Output", padding="10")
        output_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Create chart frame
        self.chart_frame = ttk.Frame(output_frame)
        self.chart_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create summary text
        self.summary_text = tk.Text(output_frame, height=10, width=50)
        self.summary_text.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

    def set_save_location(self):
        """Allow user to set default save location"""
        new_dir = filedialog.askdirectory(
            initialdir=self.config['save_dir'],
            title="Select Default Save Location"
        )
        if new_dir:
            self.config['save_dir'] = Path(new_dir)
            messagebox.showinfo("Success", 
                              f"Default save location set to:\n{new_dir}")

    def save_plot(self):
        """Save the current plot"""
        try:
            ticker = self.ticker_entry.get().upper()
            if not ticker:
                messagebox.showerror("Error", "Please run analysis first")
                return
            
            # Ensure save directory exists
            self.config['save_dir'].mkdir(exist_ok=True)
            
            # Default filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_filename = f"{ticker}_analysis_{timestamp}.png"
            
            # Open file dialog
            filename = filedialog.asksaveasfilename(
                initialdir=self.config['save_dir'],
                initialfile=default_filename,
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                # Save with high quality
                plt.savefig(filename, 
                          dpi=self.config['default_dpi'],
                          bbox_inches='tight',
                          facecolor='white',
                          edgecolor='none',
                          transparent=False)
                
                messagebox.showinfo("Success", 
                                  f"Plot saved as:\n{filename}")
                
                # Offer to open containing folder
                if messagebox.askyesno("Open Folder",
                                     "Would you like to open the containing folder?"):
                    os.startfile(os.path.dirname(filename))
            
        except Exception as e:
            messagebox.showerror("Error", 
                               f"Failed to save plot: {str(e)}")
            
        finally:
            # Ensure plot memory is cleared
            plt.close('all')

    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        # Calculate EMAs
        self.data['EMA9'] = self.data['Close'].ewm(span=9, adjust=False).mean()
        self.data['EMA13'] = self.data['Close'].ewm(span=13, adjust=False).mean()
        self.data['EMA20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
        self.data['EMA50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        self.data['EMA100'] = self.data['Close'].ewm(span=100, adjust=False).mean()
        self.data['EMA200'] = self.data['Close'].ewm(span=200, adjust=False).mean()
        
        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # ... rest of existing indicators ...

    def plot_technical_analysis(self, save_path=None):
        """Create technical analysis plot"""
        # Create figure and subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # Price and EMAs plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.data.index, self.data['Close'], label='Price', color='black')
        ax1.plot(self.data.index, self.data['EMA9'], label='EMA9', linewidth=1)
        ax1.plot(self.data.index, self.data['EMA13'], label='EMA13', linewidth=1)
        ax1.plot(self.data.index, self.data['EMA20'], label='EMA20', linewidth=1)
        ax1.plot(self.data.index, self.data['EMA50'], label='EMA50', linewidth=1)
        ax1.plot(self.data.index, self.data['EMA100'], label='EMA100', linewidth=1)
        ax1.plot(self.data.index, self.data['EMA200'], label='EMA200', linewidth=1)
        ax1.set_title(f'{self.ticker} Technical Analysis')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Volume plot
        ax2 = fig.add_subplot(gs[1])
        ax2.bar(self.data.index, self.data['Volume'], color='gray')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # RSI plot
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)  # Overbought line
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)  # Oversold line
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True)
        ax3.legend()
        
        # MACD plot
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(self.data.index, self.data['MACD'], label='MACD')
        ax4.plot(self.data.index, self.data['Signal_Line'], label='Signal')
        ax4.bar(self.data.index, self.data['MACD_Histogram'], color='gray', alpha=0.3)
        ax4.set_ylabel('MACD')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def get_analysis_summary(self):
        """Generate analysis summary"""
        current_price = self.data['Close'][-1]
        ema9 = self.data['EMA9'][-1]
        ema13 = self.data['EMA13'][-1]
        ema20 = self.data['EMA20'][-1]
        ema50 = self.data['EMA50'][-1]
        rsi = self.data['RSI'][-1]
        macd = self.data['MACD'][-1]
        signal = self.data['Signal_Line'][-1]
        
        summary = {
            'Current Price': round(current_price, 2),
            'EMA9': round(ema9, 2),
            'EMA13': round(ema13, 2),
            'EMA20': round(ema20, 2),
            'EMA50': round(ema50, 2),
            'RSI': round(rsi, 2),
            'MACD': round(macd, 2),
            'MACD Signal': round(signal, 2),
            'Trend': 'Bullish' if current_price > ema50 else 'Bearish',
            'RSI Status': 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral',
            'MACD Signal': 'Buy' if macd > signal else 'Sell',
            'Short-term Trend': 'Bullish' if ema9 > ema13 else 'Bearish'
        }
        
        return summary

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
            
            # Configure plot style
            plt.style.use('classic')
            plt.rcParams['figure.figsize'] = [15, 12]
            
            # Get stock data
            stock = yf.Ticker(ticker)
            self.data = stock.history(start=self.start_date.get(), end=self.end_date.get())
            
            # Calculate EMAs
            self.data['EMA9'] = self.data['Close'].ewm(span=9, adjust=False).mean()
            self.data['EMA13'] = self.data['Close'].ewm(span=13, adjust=False).mean()
            self.data['EMA20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
            self.data['EMA50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
            self.data['EMA100'] = self.data['Close'].ewm(span=100, adjust=False).mean()
            self.data['EMA200'] = self.data['Close'].ewm(span=200, adjust=False).mean()
            
            # Calculate RSI
            delta = self.data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # Create figure and subplots
            fig = plt.figure()
            gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
            
            # Price and EMAs plot
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(self.data.index, self.data['Close'], label='Price', color='black', linewidth=1)
            ax1.plot(self.data.index, self.data['EMA9'], label='EMA9', linewidth=1)
            ax1.plot(self.data.index, self.data['EMA13'], label='EMA13', linewidth=1)
            ax1.plot(self.data.index, self.data['EMA20'], label='EMA20', linewidth=1)
            ax1.plot(self.data.index, self.data['EMA50'], label='EMA50', linewidth=1)
            ax1.plot(self.data.index, self.data['EMA100'], label='EMA100', linewidth=1)
            ax1.plot(self.data.index, self.data['EMA200'], label='EMA200', linewidth=1)
            ax1.set_title(f'{ticker} Technical Analysis')
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax1.grid(True)
            
            # Volume plot
            ax2 = fig.add_subplot(gs[1])
            ax2.bar(self.data.index, self.data['Volume'], color='gray', alpha=0.5)
            ax2.set_ylabel('Volume')
            ax2.grid(True)
            
            # RSI plot
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(self.data.index, self.data['RSI'], label='RSI', color='purple', linewidth=1)
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax3.fill_between(self.data.index, 70, self.data['RSI'], 
                           where=self.data['RSI'] >= 70, color='r', alpha=0.3)
            ax3.fill_between(self.data.index, 30, self.data['RSI'], 
                           where=self.data['RSI'] <= 30, color='g', alpha=0.3)
            ax3.set_ylabel('RSI')
            ax3.set_ylim(0, 100)
            ax3.grid(True)
            ax3.legend()
            
            # MACD plot
            exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(self.data.index, macd, label='MACD', linewidth=1)
            ax4.plot(self.data.index, signal, label='Signal', linewidth=1)
            ax4.bar(self.data.index, hist, color='gray', alpha=0.3)
            ax4.set_ylabel('MACD')
            ax4.grid(True)
            ax4.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Display chart
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            
            # Display summary
            self.summary_text.insert(tk.END, "Technical Analysis Summary:\n\n")
            self.summary_text.insert(tk.END, f"Current Price: {self.data['Close'][-1]:.2f}\n")
            self.summary_text.insert(tk.END, f"RSI (14): {self.data['RSI'][-1]:.2f}\n")
            self.summary_text.insert(tk.END, f"EMA9: {self.data['EMA9'][-1]:.2f}\n")
            self.summary_text.insert(tk.END, f"EMA13: {self.data['EMA13'][-1]:.2f}\n")
            self.summary_text.insert(tk.END, f"EMA20: {self.data['EMA20'][-1]:.2f}\n")
            self.summary_text.insert(tk.END, f"EMA50: {self.data['EMA50'][-1]:.2f}\n")
            self.summary_text.insert(tk.END, f"EMA100: {self.data['EMA100'][-1]:.2f}\n")
            self.summary_text.insert(tk.END, f"EMA200: {self.data['EMA200'][-1]:.2f}\n")
            
            # Add RSI interpretation
            rsi_value = self.data['RSI'][-1]
            if rsi_value > 70:
                rsi_status = "Overbought"
            elif rsi_value < 30:
                rsi_status = "Oversold"
            else:
                rsi_status = "Neutral"
            self.summary_text.insert(tk.END, f"\nRSI Status: {rsi_status}\n")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            print(traceback.format_exc())  # Print detailed error for debugging

    def clear_output(self):
        """Clear the output section"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        self.summary_text.delete(1.0, tk.END)

    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    app = StockAnalyzerUI()
    app.run()

if __name__ == "__main__":
    main() 