import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Optional

class StockAnalyzer:
    def get_earnings_dates(self, ticker: str) -> list:
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            if calendar is not None and not calendar.empty:
                return [calendar.index[0]]  # Return the next earnings date
            return []
        except Exception:
            return []

    def analyze_earnings_impact(self, ticker: str, er_date: datetime, window_days: int) -> Optional[Dict]:
        try:
            # Get stock data
            end_date = er_date + timedelta(days=window_days)
            start_date = er_date - timedelta(days=window_days)
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            # Calculate returns
            data['Daily_Return'] = data['Close'].pct_change()
            data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
            
            return {'data': data}
        except Exception:
            return None

class StockAnalyzerUI:
    def __init__(self):
        self.analyzer = StockAnalyzer()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Stock Earnings Analyzer")
        self.root.geometry("800x600")
        
        # Create input frame
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.pack(fill=tk.X)
        
        # Ticker input
        ttk.Label(input_frame, text="Ticker:").pack(side=tk.LEFT)
        self.ticker_entry = ttk.Entry(input_frame, width=10)
        self.ticker_entry.pack(side=tk.LEFT, padx=5)
        
        # Analyze button
        ttk.Button(input_frame, text="Analyze", command=self.analyze_stock).pack(side=tk.LEFT, padx=5)
        
        # Create plot frame
        self.plot_frame = ttk.Frame(self.root, padding="10")
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def analyze_stock(self):
        ticker = self.ticker_entry.get().upper()
        if not ticker:
            messagebox.showerror("Error", "Please enter a ticker symbol")
            return
            
        # Get earnings dates
        dates = self.analyzer.get_earnings_dates(ticker)
        if not dates:
            messagebox.showerror("Error", f"No earnings dates found for {ticker}")
            return
            
        # Analyze most recent earnings
        latest_er = dates[0]
        analysis = self.analyzer.analyze_earnings_impact(ticker, latest_er, 30)
        if not analysis:
            messagebox.showerror("Error", f"Could not analyze {ticker}")
            return
            
        # Plot the results
        self.plot_analysis(analysis, ticker, latest_er)
    
    def plot_analysis(self, analysis, ticker, er_date):
        self.ax.clear()
        
        data = analysis['data']
        data['Cumulative_Return'].plot(ax=self.ax)
        
        self.ax.axvline(x=er_date, color='r', linestyle='--', label='Earnings Date')
        self.ax.set_title(f"{ticker} Returns Around {er_date.strftime('%Y-%m-%d')} Earnings")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Cumulative Return")
        self.ax.legend()
        self.ax.grid(True)
        
        self.canvas.draw()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = StockAnalyzerUI()
    app.run()