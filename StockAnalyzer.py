import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class StockAnalyzer:
    def __init__(self, ticker, start_date=None, end_date=None):
        """
        Initialize StockAnalyzer with ticker and date range
        
        Parameters:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format (default: 1 year ago)
        end_date (str): End date in 'YYYY-MM-DD' format (default: today)
        """
        self.ticker = ticker
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date if start_date else (
            datetime.strptime(self.end_date, '%Y-%m-%d') - timedelta(days=365)
        ).strftime('%Y-%m-%d')
        self.data = self._get_stock_data()
        
    def _get_stock_data(self):
        """Fetch stock data from Yahoo Finance"""
        stock = yf.Ticker(self.ticker)
        data = stock.history(start=self.start_date, end=self.end_date)
        return data
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        # Calculate EMAs
        self.data['EMA20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
        self.data['EMA50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        self.data['EMA100'] = self.data['Close'].ewm(span=100, adjust=False).mean()
        self.data['EMA200'] = self.data['Close'].ewm(span=200, adjust=False).mean()
        
        # Calculate MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal_Line']
        
        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Accumulation/Distribution
        clv = ((self.data['Close'] - self.data['Low']) - 
               (self.data['High'] - self.data['Close'])) / (self.data['High'] - self.data['Low'])
        self.data['ADL'] = clv * self.data['Volume']
        self.data['ADL'] = self.data['ADL'].cumsum()
        
        return self.data
    
    def plot_technical_analysis(self, save_path=None):
        """
        Create technical analysis plot
        
        Parameters:
        save_path (str): Path to save the plot (optional)
        """
        # Create figure and subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        
        # Price and EMAs plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.data.index, self.data['Close'], label='Price', color='black')
        ax1.plot(self.data.index, self.data['EMA20'], label='EMA20')
        ax1.plot(self.data.index, self.data['EMA50'], label='EMA50')
        ax1.plot(self.data.index, self.data['EMA100'], label='EMA100')
        ax1.plot(self.data.index, self.data['EMA200'], label='EMA200')
        ax1.set_title(f'{self.ticker} Technical Analysis')
        ax1.legend()
        ax1.grid(True)
        
        # Volume plot
        ax2 = fig.add_subplot(gs[1])
        ax2.bar(self.data.index, self.data['Volume'], color='gray')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # MACD plot
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(self.data.index, self.data['MACD'], label='MACD')
        ax3.plot(self.data.index, self.data['Signal_Line'], label='Signal')
        ax3.bar(self.data.index, self.data['MACD_Histogram'], color='gray', alpha=0.3)
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def get_analysis_summary(self):
        """Generate analysis summary"""
        current_price = self.data['Close'][-1]
        ema20 = self.data['EMA20'][-1]
        ema50 = self.data['EMA50'][-1]
        rsi = self.data['RSI'][-1]
        macd = self.data['MACD'][-1]
        signal = self.data['Signal_Line'][-1]
        
        summary = {
            'Current Price': round(current_price, 2),
            'EMA20': round(ema20, 2),
            'EMA50': round(ema50, 2),
            'RSI': round(rsi, 2),
            'MACD': round(macd, 2),
            'MACD Signal': round(signal, 2),
            'Trend': 'Bullish' if current_price > ema50 else 'Bearish',
            'RSI Status': 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral',
            'MACD Signal': 'Buy' if macd > signal else 'Sell'
        }
        
        return summary

def analyze_stock(ticker, start_date=None, end_date=None, save_plot=None):
    """
    Convenience function to analyze a stock
    
    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    save_plot (str): Path to save the plot
    
    Returns:
    tuple: (analysis_summary, figure)
    """
    analyzer = StockAnalyzer(ticker, start_date, end_date)
    analyzer.calculate_technical_indicators()
    summary = analyzer.get_analysis_summary()
    fig = analyzer.plot_technical_analysis(save_plot)
    return summary, fig

# Example usage:
if __name__ == "__main__":
    # Analyze Tesla stock
    summary, fig = analyze_stock('TSLA', 
                               start_date='2024-01-01',
                               save_plot='tesla_analysis.png')
    
    print("\nTechnical Analysis Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    plt.show()