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
        self.cache = {}
        self.current_results = {}
        self.current_er_date = None

    def get_earnings_dates(self, ticker: str) -> List[datetime]:
        """Fetch historical earnings dates for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            earnings_history = stock.earnings_dates
            
            if earnings_history is None or earnings_history.empty:
                return []
                
            dates = earnings_history.index
            if dates.tz is not None:
                dates = dates.tz_localize(None)
            
            return sorted(dates, reverse=True)[:8]
            
        except Exception as e:
            print(f"Error fetching earnings dates for {ticker}: {e}")
            return []

    def get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get stock price data with caching"""
        try:
            start_date = pd.to_datetime(start_date).tz_localize(None)
            extended_start = start_date - timedelta(days=400)
            end_date = pd.to_datetime(end_date).tz_localize(None)
            
            cache_key = f"{ticker}_{start_date}_{end_date}"
            if cache_key in self.cache:
                return self.cache[cache_key]
                
            stock = yf.Ticker(ticker)
            data = stock.history(start=extended_start, end=end_date)
            
            if not data.empty:
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                
                # Calculate all metrics
                data['Daily_Return'] = data['Close'].pct_change()
                data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
                
                # RSI
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                # Moving Averages
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['MA200'] = data['Close'].rolling(window=200).mean()
                
                # Historical Volatility
                data['Historical_Vol'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252) * 100
                
                # Get IV if available
                try:
                    options = stock.options
                    if options:
                        nearest_option = stock.option_chain(options[0])
                        data['IV'] = nearest_option.calls['impliedVolatility'].mean() * 100
                    else:
                        data['IV'] = None
                except:
                    data['IV'] = None
                
                data = data[data.index >= start_date]
                self.cache[cache_key] = data
                return data
                
            return None
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def get_price_levels(self, ticker: str) -> Dict:
        """Get price levels including 52-week and all-time highs"""
        try:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period="max")
            
            current_price = hist_data['Close'][-1]
            all_time_high = hist_data['High'].max()
            all_time_high_date = hist_data['High'].idxmax()
            
            year_data = hist_data.last('52W')
            week_high = year_data['High'].max()
            week_high_date = year_data['High'].idxmax()
            
            return {
                'Current Price': current_price,
                '52-Week High': week_high,
                '52-Week High Date': week_high_date,
                'All-Time High': all_time_high,
                'All-Time High Date': all_time_high_date,
                'Pct From 52-Week High': ((week_high - current_price) / current_price) * 100,
                'Pct From All-Time High': ((all_time_high - current_price) / current_price) * 100
            }
        except Exception as e:
            print(f"Error getting price levels for {ticker}: {e}")
            return None 