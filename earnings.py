import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
from docx import Document
from docx.shared import Inches
import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class UnifiedAnalyzer:
    def __init__(self):
        pass
        
    def get_earnings_dates(self, ticker: str) -> List[datetime]:
        stock = yf.Ticker(ticker)
        try:
            # First try: Get earnings from quarterly earnings data
            earnings = stock.quarterly_earnings
            if earnings is not None and not earnings.empty:
                return sorted(earnings.index, reverse=True)[:12]
            
            # Second try: Get earnings from earnings calendar
            calendar = stock.earnings_dates
            if calendar is not None and not calendar.empty:
                return sorted(calendar.index, reverse=True)[:12]
            
            # Third try: Get earnings from quarterly financials
            financials = stock.quarterly_financials
            if financials is not None and not financials.empty:
                return sorted(financials.columns, reverse=True)[:12]
            
            # Fourth try: Get earnings history
            history = stock.earnings_history
            if history is not None and not history.empty:
                return sorted(history.index, reverse=True)[:12]
            
            print(f"No earnings data found for {ticker}")
            return []
            
        except Exception as e:
            print(f"Error getting earnings dates for {ticker}: {e}")
            return []
        
    def get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        stock = yf.Ticker(ticker)
        try:
            # Get 3 years of data to ensure we have enough history
            return stock.history(period="3y")
        except Exception as e:
            print(f"Error getting stock data: {e}")
            return None

class UnifiedAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stock Analysis Dashboard")
        self.root.geometry("1200x800")
        
        # Initialize analyzer
        self.analyzer = UnifiedAnalyzer()
        
        # Create main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.pack(fill='both', expand=True)
        
        # Create and setup tabs
        self.setup_tabs()
        
    def setup_tabs(self):
        # Create notebook
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.earnings_tab = ttk.Frame(self.notebook)
        self.options_tab = ttk.Frame(self.notebook)
        self.price_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.earnings_tab, text='Earnings Analysis')
        self.notebook.add(self.options_tab, text='Options Analysis')
        self.notebook.add(self.price_tab, text='Price Levels')
        
        # Setup each tab
        self.setup_earnings_tab()
        self.setup_options_tab()
        self.setup_price_tab()
        
    def setup_earnings_tab(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.earnings_tab, text="Input", padding="5")
        input_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(input_frame, text="Ticker:").pack(side='left', padx=5)
        self.earnings_ticker = ttk.Entry(input_frame, width=10)
        self.earnings_ticker.pack(side='left', padx=5)
        
        ttk.Button(input_frame, text="Analyze", 
                  command=self.run_earnings_analysis).pack(side='left', padx=5)
        
        # Results frame
        self.earnings_results = ttk.LabelFrame(self.earnings_tab, text="Results", padding="5")
        self.earnings_results.pack(fill='both', expand=True, padx=5, pady=5)
        
    def setup_options_tab(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.options_tab, text="Input", padding="5")
        input_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(input_frame, text="Ticker:").pack(side='left', padx=5)
        self.options_ticker = ttk.Entry(input_frame, width=10)
        self.options_ticker.pack(side='left', padx=5)
        
        ttk.Label(input_frame, text="Analysis:").pack(side='left', padx=5)
        self.options_analysis = ttk.Combobox(input_frame, 
                                           values=["IV Analysis", "Options Chain"],
                                           state='readonly', width=15)
        self.options_analysis.pack(side='left', padx=5)
        self.options_analysis.set("IV Analysis")
        
        ttk.Button(input_frame, text="Analyze", 
                  command=self.run_options_analysis).pack(side='left', padx=5)
        
        # Results frame
        self.options_results = ttk.LabelFrame(self.options_tab, text="Results", padding="5")
        self.options_results.pack(fill='both', expand=True, padx=5, pady=5)
        
    def setup_price_tab(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.price_tab, text="Input", padding="5")
        input_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(input_frame, text="Ticker:").pack(side='left', padx=5)
        self.price_ticker = ttk.Entry(input_frame, width=10)
        self.price_ticker.pack(side='left', padx=5)
        
        ttk.Button(input_frame, text="Analyze", 
                  command=self.run_price_analysis).pack(side='left', padx=5)
        
        # Results frame
        self.price_results = ttk.LabelFrame(self.price_tab, text="Results", padding="5")
        self.price_results.pack(fill='both', expand=True, padx=5, pady=5)

    def run_earnings_analysis(self):
        ticker = self.earnings_ticker.get().strip().upper()
        if not ticker:
            messagebox.showerror("Error", "Please enter a ticker symbol")
            return
            
        try:
            # Clear previous results
            for widget in self.earnings_results.winfo_children():
                widget.destroy()
                
            # Get earnings dates
            dates = self.analyzer.get_earnings_dates(ticker)
            if not dates:
                messagebox.showwarning("Warning", "No earnings dates found")
                return
                
            # Get historical data
            stock = yf.Ticker(ticker)
            data = stock.history(period="max")  # Get maximum available history
            
            if data.empty:
                messagebox.showerror("Error", "Could not retrieve stock data")
                return
                
            # Create treeview with more detailed price columns
            tree = ttk.Treeview(self.earnings_results, columns=(
                "Date", "Pre-5d", "Pre-3d", "Pre-1d", 
                "Post-1d", "Post-3d", "Post-5d", "Change"
            ))
            
            # Configure column headings and hide default first column
            tree["show"] = "headings"
            tree.heading("Date", text="ER Date")
            tree.heading("Pre-5d", text="5d Before")
            tree.heading("Pre-3d", text="3d Before")
            tree.heading("Pre-1d", text="1d Before")
            tree.heading("Post-1d", text="1d After")
            tree.heading("Post-3d", text="3d After")
            tree.heading("Post-5d", text="5d After")
            tree.heading("Change", text="5d % Change")
            
            # Configure column widths
            for col in tree["columns"]:
                tree.column(col, width=100)
            
            tree.pack(fill='both', expand=True)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(self.earnings_results, orient="vertical", command=tree.yview)
            scrollbar.pack(side='right', fill='y')
            tree.configure(yscrollcommand=scrollbar.set)
            
            # Process each earnings date
            for date in dates:
                try:
                    # Get 5 days before and after earnings
                    start_date = date - timedelta(days=7)
                    end_date = date + timedelta(days=7)
                    
                    mask = (data.index >= start_date) & (data.index <= end_date)
                    period_data = data[mask]
                    
                    if len(period_data) >= 6:
                        pre_5d = period_data['Close'].iloc[0]
                        pre_3d = period_data['Close'].iloc[1]
                        pre_1d = period_data['Close'].iloc[2]
                        post_1d = period_data['Close'].iloc[3]
                        post_3d = period_data['Close'].iloc[4]
                        post_5d = period_data['Close'].iloc[5]
                        
                        change = ((post_5d - pre_5d) / pre_5d) * 100
                        
                        tree.insert("", "end", values=(
                            date.strftime("%Y-%m-%d"),
                            f"${pre_5d:.2f}",
                            f"${pre_3d:.2f}",
                            f"${pre_1d:.2f}",
                            f"${post_1d:.2f}",
                            f"${post_3d:.2f}",
                            f"${post_5d:.2f}",
                            f"{change:.2f}%"
                        ))
                        
                except Exception as e:
                    print(f"Error processing date {date}: {e}")
                    continue
                    
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_options_analysis(self):
        ticker = self.options_ticker.get().strip().upper()
        if not ticker:
            messagebox.showerror("Error", "Please enter a ticker symbol")
            return
            
        try:
            # Clear previous results
            for widget in self.options_results.winfo_children():
                widget.destroy()
                
            stock = yf.Ticker(ticker)
            options = stock.options
            
            if not options:
                messagebox.showwarning("Warning", "No options data available")
                return
                
            # Create treeview for options
            tree = ttk.Treeview(self.options_results, columns=(
                "Date", "Strike", "Call_Price", "Put_Price", "Volume"
            ))
            
            tree["show"] = "headings"
            tree.heading("Date", text="Expiration")
            tree.heading("Strike", text="Strike Price")
            tree.heading("Call_Price", text="Call Price")
            tree.heading("Put_Price", text="Put Price")
            tree.heading("Volume", text="Volume")
            
            for col in tree["columns"]:
                tree.column(col, width=100)
            
            tree.pack(fill='both', expand=True)
            
            # Get first expiration date's options
            chain = stock.option_chain(options[0])
            
            # Display first 10 strikes
            for i in range(min(10, len(chain.calls))):
                call = chain.calls.iloc[i]
                put = chain.puts.iloc[i]
                tree.insert("", "end", values=(
                    options[0],
                    f"${call['strike']:.2f}",
                    f"${call['lastPrice']:.2f}",
                    f"${put['lastPrice']:.2f}",
                    call['volume']
                ))
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_price_analysis(self):
        ticker = self.price_ticker.get().strip().upper()
        if not ticker:
            messagebox.showerror("Error", "Please enter a ticker symbol")
            return
            
        try:
            # Clear previous results
            for widget in self.price_results.winfo_children():
                widget.destroy()
                
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            # Calculate key levels
            current_price = hist['Close'].iloc[-1]
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            ma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # Create result frame
            result_frame = ttk.Frame(self.price_results)
            result_frame.pack(fill='both', expand=True)
            
            # Display price levels
            ttk.Label(result_frame, text=f"Current Price: ${current_price:.2f}").pack()
            ttk.Label(result_frame, text=f"52-Week High: ${high_52w:.2f}").pack()
            ttk.Label(result_frame, text=f"52-Week Low: ${low_52w:.2f}").pack()
            ttk.Label(result_frame, text=f"50-Day MA: ${ma_50:.2f}").pack()
            ttk.Label(result_frame, text=f"200-Day MA: ${ma_200:.2f}").pack()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = UnifiedAnalyzerGUI()
    app.run()