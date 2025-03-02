import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

class ZMTechAnalysis:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else Path('zmtech_finance/data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Configure ZMTech plotting style"""
        plt.style.use('default')
        self.colors = {
            'primary': '#1E3D59',
            'secondary': '#17B890',
            'accent': '#FFB400',
            'background': '#E8E8E8'
        }
        
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def analyze_stocks(self, ticker1, ticker2, periods=10):  # This is the correct method name
        """Perform comprehensive stock analysis"""
        try:
            # Get stock data
            stock1 = yf.download(ticker1, start='2022-01-01')
            stock2 = yf.download(ticker2, start='2022-01-01')
            
            # Basic analysis
            analysis = {
                'technical': {
                    'stock1': {
                        'price': stock1['Close'][-1],
                        'volume': stock1['Volume'][-1],
                        'change': ((stock1['Close'][-1] / stock1['Close'][-2]) - 1) * 100
                    },
                    'stock2': {
                        'price': stock2['Close'][-1],
                        'volume': stock2['Volume'][-1],
                        'change': ((stock2['Close'][-1] / stock2['Close'][-2]) - 1) * 100
                    }
                },
                'correlation': {
                    'correlation': stock1['Close'].corr(stock2['Close'])
                },
                'charts': self.generate_charts(stock1, stock2)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return None

    def generate_charts(self, stock1_data, stock2_data):
        """Generate analysis charts"""
        charts = {}
        
        # Price comparison chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock1_data.index, stock1_data['Close'], 
                label='Stock 1', color=self.colors['primary'])
        ax.plot(stock2_data.index, stock2_data['Close'],
                label='Stock 2', color=self.colors['secondary'])
        ax.set_title('Price Comparison')
        ax.legend()
        charts['price_comparison'] = fig
        
        # Volume chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(stock1_data.index, stock1_data['Volume'],
               alpha=0.5, color=self.colors['primary'], label='Stock 1')
        ax.bar(stock2_data.index, stock2_data['Volume'],
               alpha=0.5, color=self.colors['secondary'], label='Stock 2')
        ax.set_title('Volume Comparison')
        ax.legend()
        charts['volume_comparison'] = fig
        
        return charts

def main():
    # Test the analysis engine
    analyzer = ZMTechAnalysis()
    results = analyzer.analyze_stocks('AAPL', 'MSFT')
    print("Analysis completed")

if __name__ == "__main__":
    main()